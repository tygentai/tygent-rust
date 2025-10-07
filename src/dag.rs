use std::collections::{HashMap, HashSet, VecDeque};

use crate::nodes::{Node, NodeInputs, NodeResult};
use crate::scheduler::{Scheduler, SchedulerError};
use thiserror::Error;

pub type EdgeMetadata = HashMap<String, String>;
pub type EdgeMetadataMap = HashMap<String, HashMap<String, EdgeMetadata>>;
pub type NodeOutputs = HashMap<String, NodeResult>;

#[derive(Debug, Error)]
pub enum DagError {
    #[error("source node `{0}` not found in DAG")]
    SourceMissing(String),
    #[error("target node `{0}` not found in DAG")]
    TargetMissing(String),
    #[error("cycle detected in DAG; remaining nodes: {remaining:?}")]
    CycleDetected { remaining: Vec<String> },
}

#[derive(Default)]
pub struct Dag {
    name: String,
    nodes: HashMap<String, Box<dyn Node>>,
    edges: HashMap<String, Vec<String>>,
    edge_mappings: EdgeMetadataMap,
}

impl Clone for Dag {
    fn clone(&self) -> Self {
        self.copy()
    }
}

impl Dag {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn nodes(&self) -> impl Iterator<Item = (&String, &Box<dyn Node>)> {
        self.nodes.iter()
    }

    pub fn edges(&self) -> &HashMap<String, Vec<String>> {
        &self.edges
    }

    pub fn edge_mappings(&self) -> &EdgeMetadataMap {
        &self.edge_mappings
    }

    pub fn add_node<N>(&mut self, node: N)
    where
        N: Node + 'static,
    {
        let name = node.name().to_string();
        self.nodes.insert(name, Box::new(node));
    }

    pub fn add_boxed_node(&mut self, node: Box<dyn Node>) {
        let name = node.name().to_string();
        self.nodes.insert(name, node);
    }

    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    pub fn get_node(&self, name: &str) -> Option<&dyn Node> {
        self.nodes.get(name).map(|node| node.as_ref())
    }

    pub fn add_edge(
        &mut self,
        from_node: &str,
        to_node: &str,
        metadata: Option<EdgeMetadata>,
    ) -> Result<(), DagError> {
        if !self.nodes.contains_key(from_node) {
            return Err(DagError::SourceMissing(from_node.to_string()));
        }
        if !self.nodes.contains_key(to_node) {
            return Err(DagError::TargetMissing(to_node.to_string()));
        }

        let entry = self
            .edges
            .entry(from_node.to_string())
            .or_insert_with(Vec::new);
        if !entry.contains(&to_node.to_string()) {
            entry.push(to_node.to_string());
        }

        if let Some(target) = self.nodes.get_mut(to_node) {
            let mut dependencies = target.dependencies();
            if !dependencies.contains(&from_node.to_string()) {
                dependencies.push(from_node.to_string());
                target.set_dependencies(dependencies);
            }
        }

        if let Some(meta) = metadata {
            self.edge_mappings
                .entry(from_node.to_string())
                .or_insert_with(HashMap::new)
                .insert(to_node.to_string(), meta);
        }

        Ok(())
    }

    pub fn get_topological_order(&self) -> Result<Vec<String>, DagError> {
        let mut incoming: HashMap<String, HashSet<String>> = self
            .nodes
            .keys()
            .map(|name| (name.clone(), HashSet::new()))
            .collect();

        for (source, targets) in &self.edges {
            for target in targets {
                if let Some(set) = incoming.get_mut(target) {
                    set.insert(source.clone());
                }
            }
        }

        let mut queue: VecDeque<String> = incoming
            .iter()
            .filter_map(|(name, deps)| {
                if deps.is_empty() {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.clone());

            if let Some(children) = self.edges.get(&node) {
                for child in children {
                    if let Some(deps) = incoming.get_mut(child) {
                        deps.remove(&node);
                        if deps.is_empty() {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            let remaining: Vec<String> = self
                .nodes
                .keys()
                .filter(|name| !order.contains(name))
                .cloned()
                .collect();
            return Err(DagError::CycleDetected { remaining });
        }

        Ok(order)
    }

    pub fn get_roots_and_leaves(&self) -> (Vec<String>, Vec<String>) {
        let roots = self
            .nodes
            .iter()
            .filter_map(|(name, node)| {
                if node.dependencies().is_empty() {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let leaves = self
            .nodes
            .keys()
            .filter(|name| self.edges.get(*name).map(Vec::is_empty).unwrap_or(true))
            .cloned()
            .collect();

        (roots, leaves)
    }

    pub fn copy(&self) -> Self {
        let nodes = self
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.clone()))
            .collect();

        Self {
            name: self.name.clone(),
            nodes,
            edges: self.edges.clone(),
            edge_mappings: self.edge_mappings.clone(),
        }
    }

    pub fn compute_critical_path(&self) -> Result<HashMap<String, f64>, DagError> {
        let mut cp: HashMap<String, f64> = self
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.latency_estimate()))
            .collect();

        let order = self.get_topological_order()?;

        for node_name in order.into_iter().rev() {
            if let Some(children) = self.edges.get(&node_name) {
                if children.is_empty() {
                    continue;
                }

                let max_child = children
                    .iter()
                    .filter_map(|child| cp.get(child))
                    .fold(0.0_f64, |acc, value| acc.max(*value));

                if let Some(node) = self.nodes.get(&node_name) {
                    let updated = node.latency_estimate() + max_child;
                    cp.insert(node_name, updated);
                }
            }
        }

        Ok(cp)
    }

    pub async fn execute(&self, inputs: NodeInputs) -> Result<NodeOutputs, SchedulerError> {
        let mut scheduler = Scheduler::new(self.copy());
        let results = scheduler.execute(inputs).await?;
        Ok(results.results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{NodeInputs, ToolNode};
    use serde_json::{json, Value};

    fn tool(name: &str) -> ToolNode {
        ToolNode::new_sync(name.to_string(), |_inputs| Ok(json!(null)))
    }

    #[test]
    fn add_nodes_and_edges_updates_dependencies() {
        let mut dag = Dag::new("test");
        dag.add_node(tool("A"));
        dag.add_node(tool("B"));
        dag.add_edge("A", "B", None).unwrap();

        let deps = dag.get_node("B").unwrap().dependencies();
        assert_eq!(deps, vec!["A".to_string()]);
        assert_eq!(dag.edges().get("A").unwrap(), &vec!["B".to_string()]);
    }

    #[test]
    fn duplicate_edges_are_ignored() {
        let mut dag = Dag::new("test");
        dag.add_node(tool("A"));
        dag.add_node(tool("B"));
        dag.add_edge("A", "B", None).unwrap();
        dag.add_edge("A", "B", None).unwrap();
        assert_eq!(dag.edges().get("A").unwrap().len(), 1);
    }

    #[test]
    fn topological_order_matches_expectations() {
        let mut dag = Dag::new("plan");
        dag.add_node(tool("start"));
        dag.add_node(tool("middle"));
        dag.add_node(tool("end"));
        dag.add_edge("start", "middle", None).unwrap();
        dag.add_edge("middle", "end", None).unwrap();

        let order = dag.get_topological_order().unwrap();
        assert_eq!(order, vec!["start", "middle", "end"]);
    }

    #[test]
    fn cycle_detection_returns_error() {
        let mut dag = Dag::new("cycle");
        dag.add_node(tool("A"));
        dag.add_node(tool("B"));
        dag.add_edge("A", "B", None).unwrap();
        dag.add_edge("B", "A", None).unwrap();

        let err = dag.get_topological_order().unwrap_err();
        matches!(err, DagError::CycleDetected { .. });
    }

    #[test]
    fn roots_and_leaves_identified() {
        let mut dag = Dag::new("roots_leaves");
        dag.add_node(tool("root"));
        dag.add_node(tool("branch"));
        dag.add_node(tool("leaf"));
        dag.add_edge("root", "branch", None).unwrap();
        dag.add_edge("branch", "leaf", None).unwrap();

        let (roots, leaves) = dag.get_roots_and_leaves();
        assert_eq!(roots, vec!["root"]);
        assert_eq!(leaves, vec!["leaf"]);
    }

    #[test]
    fn copy_creates_independent_structure() {
        let mut dag = Dag::new("copy");
        dag.add_node(tool("A"));
        dag.add_node(tool("B"));
        dag.add_edge("A", "B", None).unwrap();

        let mut cloned = dag.copy();
        cloned.add_node(tool("C"));
        cloned.add_edge("B", "C", None).unwrap();

        assert!(cloned.has_node("C"));
        assert!(!dag.has_node("C"));
        assert_eq!(dag.edges().get("B"), None);
    }

    #[test]
    fn critical_path_accumulates_latencies() {
        let mut dag = Dag::new("latency");
        dag.add_node(tool("A").with_latency_estimate(1.0));
        dag.add_node(tool("B").with_latency_estimate(2.0));
        dag.add_node(tool("C").with_latency_estimate(4.0));
        dag.add_edge("A", "B", None).unwrap();
        dag.add_edge("B", "C", None).unwrap();

        let cp = dag.compute_critical_path().unwrap();
        assert_eq!(cp.get("C").copied().unwrap(), 4.0);
        assert_eq!(cp.get("B").copied().unwrap(), 6.0);
        assert_eq!(cp.get("A").copied().unwrap(), 7.0);
    }

    #[tokio::test]
    async fn dag_execute_runs_nodes() {
        let mut dag = Dag::new("exec");
        dag.add_node(ToolNode::new_sync("start", |_| Ok(json!({"value": 1}))));
        dag.add_node(ToolNode::new_sync("end", |inputs| {
            let start = inputs
                .get("start")
                .and_then(Value::as_object)
                .and_then(|m| m.get("value"))
                .and_then(Value::as_i64)
                .unwrap_or_default();
            Ok(json!({"total": start + 1}))
        }));
        dag.add_edge("start", "end", None).unwrap();
        let results = dag.execute(NodeInputs::new()).await.unwrap();
        assert_eq!(results["start"]["value"], 1);
        assert_eq!(results["end"]["total"], 2);
    }

    #[test]
    fn edge_metadata_is_stored() {
        let mut dag = Dag::new("meta");
        dag.add_node(tool("A"));
        dag.add_node(tool("B"));
        let mut meta = HashMap::new();
        meta.insert("label".into(), "edge".into());
        dag.add_edge("A", "B", Some(meta)).unwrap();

        let stored = dag.edge_mappings().get("A").unwrap().get("B").unwrap();
        assert_eq!(stored.get("label").unwrap(), "edge");
    }

    #[test]
    fn missing_nodes_raise_errors() {
        let mut dag = Dag::new("missing");
        dag.add_node(tool("A"));
        let err = dag.add_edge("A", "B", None).unwrap_err();
        matches!(err, DagError::TargetMissing(_));

        let err = dag.add_edge("B", "A", None).unwrap_err();
        matches!(err, DagError::SourceMissing(_));
    }
}
