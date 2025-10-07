use crate::dag::{Dag, DagError, EdgeMetadata};
use crate::nodes::{Node, ToolNode};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParsePlanError {
    #[error(transparent)]
    Dag(#[from] DagError),
    #[error("duplicate node name `{0}` in plans")]
    DuplicateNode(String),
}

#[derive(Clone)]
pub struct PlanSpec {
    pub name: Option<String>,
    pub steps: Vec<PlanStep>,
}

impl PlanSpec {
    pub fn new(name: Option<String>, steps: Vec<PlanStep>) -> Self {
        Self { name, steps }
    }
}

#[derive(Clone)]
pub struct PlanStep {
    pub node: Box<dyn Node>,
    pub dependencies: Vec<String>,
    pub critical: bool,
}

impl PlanStep {
    pub fn new<N>(node: N) -> Self
    where
        N: Node + 'static,
    {
        Self {
            node: Box::new(node),
            dependencies: Vec::new(),
            critical: false,
        }
    }

    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    pub fn critical(mut self, critical: bool) -> Self {
        self.critical = critical;
        self
    }
}

impl From<ToolNode> for PlanStep {
    fn from(node: ToolNode) -> Self {
        PlanStep::new(node)
    }
}

pub fn parse_plan(spec: PlanSpec) -> Result<(Dag, Vec<String>), ParsePlanError> {
    let name = spec.name.unwrap_or_else(|| "plan_dag".to_string());
    let mut dag = Dag::new(name);
    let mut critical = Vec::new();

    let mut dependencies: Vec<(String, Vec<String>)> = Vec::with_capacity(spec.steps.len());

    for step in spec.steps {
        let node_name = step.node.name().to_string();
        if step.critical {
            critical.push(node_name.clone());
        }
        dependencies.push((node_name.clone(), step.dependencies));
        dag.add_boxed_node(step.node);
    }

    for (node_name, deps) in dependencies {
        for dep in deps {
            dag.add_edge(&dep, &node_name, None)?;
        }
    }

    Ok((dag, critical))
}

pub fn parse_plans<I>(plans: I) -> Result<(Dag, Vec<String>), ParsePlanError>
where
    I: IntoIterator<Item = PlanSpec>,
{
    let mut merged = Dag::new("combined_plan");
    let mut critical_all = Vec::new();

    for plan in plans.into_iter() {
        let (dag, critical) = parse_plan(plan)?;

        for (name, node) in dag.nodes() {
            if merged.has_node(name) {
                return Err(ParsePlanError::DuplicateNode(name.clone()));
            }
            merged.add_boxed_node(node.clone());
        }

        for (src, targets) in dag.edges() {
            for tgt in targets {
                let meta: Option<EdgeMetadata> = dag
                    .edge_mappings()
                    .get(src)
                    .and_then(|map| map.get(tgt))
                    .cloned();
                merged.add_edge(src, tgt, meta)?;
            }
        }

        critical_all.extend(critical);
    }

    Ok((merged, critical_all))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::ToolNode;
    use serde_json::json;

    fn noop_tool(name: &str) -> ToolNode {
        ToolNode::new_sync(name.to_string(), |_inputs| Ok(json!(null)))
    }

    #[test]
    fn parse_plan_builds_dag() {
        let steps = vec![
            PlanStep::new(noop_tool("start")),
            PlanStep::new(noop_tool("end")).with_dependencies(vec!["start".into()]),
        ];
        let spec = PlanSpec::new(Some("flow".into()), steps);
        let (dag, critical) = parse_plan(spec).unwrap();
        assert!(dag.has_node("start"));
        assert!(dag.has_node("end"));
        assert_eq!(critical, Vec::<String>::new());
        let edges = dag.edges();
        assert_eq!(edges.get("start").unwrap(), &vec!["end".to_string()]);
    }

    #[test]
    fn critical_steps_are_collected() {
        let steps = vec![
            PlanStep::new(noop_tool("root")).critical(true),
            PlanStep::new(noop_tool("child")),
        ];
        let (dag, critical) = parse_plan(PlanSpec::new(None, steps)).unwrap();
        assert!(dag.has_node("root"));
        assert_eq!(critical, vec!["root".to_string()]);
    }

    #[test]
    fn missing_dependency_returns_error() {
        let steps =
            vec![PlanStep::new(noop_tool("task")).with_dependencies(vec!["missing".into()])];
        let result = parse_plan(PlanSpec::new(None, steps));
        assert!(result.is_err());
        let err = result.err().unwrap();
        matches!(
            err,
            ParsePlanError::Dag(DagError::SourceMissing(_))
                | ParsePlanError::Dag(DagError::TargetMissing(_))
        );
    }

    #[test]
    fn parse_plans_merges_multiple_specs() {
        let plan_a = PlanSpec::new(
            Some("plan_a".into()),
            vec![PlanStep::new(noop_tool("a_start"))],
        );
        let plan_b = PlanSpec::new(
            Some("plan_b".into()),
            vec![
                PlanStep::new(noop_tool("b_start")),
                PlanStep::new(noop_tool("b_end")).with_dependencies(vec!["b_start".into()]),
            ],
        );

        let (dag, critical) = parse_plans(vec![plan_a, plan_b]).unwrap();
        assert!(dag.has_node("a_start"));
        assert!(dag.has_node("b_end"));
        assert!(critical.is_empty());
        assert_eq!(
            dag.edges().get("b_start").unwrap(),
            &vec!["b_end".to_string()]
        );
    }

    #[test]
    fn parse_plans_detects_duplicate_nodes() {
        let plan_a = PlanSpec::new(None, vec![PlanStep::new(noop_tool("shared"))]);
        let plan_b = PlanSpec::new(None, vec![PlanStep::new(noop_tool("shared"))]);
        let result = parse_plans(vec![plan_a, plan_b]);
        assert!(result.is_err());
        let err = result.err().unwrap();
        matches!(err, ParsePlanError::DuplicateNode(ref name) if name == "shared");
    }
}
