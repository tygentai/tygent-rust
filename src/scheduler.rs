use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use tokio::time::timeout;

use crate::dag::{Dag, DagError, NodeOutputs};
use crate::nodes::{NodeExecutionError, NodeInputs, NodeMetadata, NodeResult};

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error(transparent)]
    Dag(#[from] DagError),
    #[error("missing node `{0}` in DAG")]
    MissingNode(String),
    #[error("deadlock detected in DAG execution. Waiting nodes: {0:?}")]
    Deadlock(Vec<String>),
    #[error("token budget exceeded when executing {0}")]
    TokenBudgetExceeded(String),
    #[error("timeout executing node `{node}` after {timeout_ms}ms")]
    Timeout { node: String, timeout_ms: u64 },
    #[error("error executing node `{node}`: {source}")]
    NodeExecution {
        node: String,
        #[source]
        source: NodeExecutionError,
    },
}

#[derive(Clone, Debug)]
pub struct ExecutionResults {
    pub results: NodeOutputs,
}

#[derive(Clone, Debug)]
pub struct SchedulerSnapshot {
    pub max_parallel_nodes: usize,
    pub max_execution_time: Duration,
    pub priority_nodes: Vec<String>,
    pub token_budget: Option<u32>,
    pub tokens_used: u32,
    pub requests_per_minute: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HookStage {
    BeforeExecute,
    AfterExecute,
    Audit,
}

#[derive(Clone, Debug)]
pub struct HookContext {
    pub stage: HookStage,
    pub node_name: String,
    pub inputs: NodeInputs,
    pub output: Option<NodeResult>,
    pub scheduler: SchedulerSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HookOutcome {
    Continue,
    Stop,
}

pub type HookFuture = std::pin::Pin<Box<dyn std::future::Future<Output = HookOutcome> + Send>>;

pub trait SchedulerHook: Send + Sync {
    fn invoke(&self, context: HookContext) -> HookFuture;
}

impl<F, Fut> SchedulerHook for F
where
    F: Send + Sync + 'static + Fn(HookContext) -> Fut,
    Fut: std::future::Future<Output = HookOutcome> + Send + 'static,
{
    fn invoke(&self, context: HookContext) -> HookFuture {
        Box::pin((self)(context))
    }
}

pub struct Scheduler {
    dag: Dag,
    audit_file: Option<PathBuf>,
    hooks: Vec<std::sync::Arc<dyn SchedulerHook>>,
    stop: bool,
    max_parallel_nodes: usize,
    max_execution_time: Duration,
    priority_nodes: Vec<String>,
    token_budget: Option<u32>,
    tokens_used: u32,
    requests_per_minute: Option<u32>,
    request_times: VecDeque<Instant>,
}

impl Scheduler {
    pub fn new(dag: Dag) -> Self {
        Self {
            dag,
            audit_file: None,
            hooks: Vec::new(),
            stop: false,
            max_parallel_nodes: 4,
            max_execution_time: Duration::from_millis(60_000),
            priority_nodes: Vec::new(),
            token_budget: None,
            tokens_used: 0,
            requests_per_minute: None,
            request_times: VecDeque::new(),
        }
    }

    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    pub fn set_audit_file(&mut self, path: Option<PathBuf>) {
        self.audit_file = path;
    }

    pub fn set_hooks(&mut self, hooks: Vec<std::sync::Arc<dyn SchedulerHook>>) {
        self.hooks = hooks;
    }

    pub fn add_hook(&mut self, hook: std::sync::Arc<dyn SchedulerHook>) {
        self.hooks.push(hook);
    }

    pub fn set_max_parallel_nodes(&mut self, max_parallel_nodes: usize) {
        self.max_parallel_nodes = max_parallel_nodes.max(1);
    }

    pub fn set_max_execution_time(&mut self, millis: u64) {
        self.max_execution_time = Duration::from_millis(millis.max(1));
    }

    pub fn set_priority_nodes(&mut self, priority_nodes: Vec<String>) {
        self.priority_nodes = priority_nodes;
    }

    pub fn set_token_budget(&mut self, budget: Option<u32>) {
        self.token_budget = budget;
    }

    pub fn set_requests_per_minute(&mut self, limit: Option<u32>) {
        self.requests_per_minute = limit;
    }

    pub fn tokens_used(&self) -> u32 {
        self.tokens_used
    }

    fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            max_parallel_nodes: self.max_parallel_nodes,
            max_execution_time: self.max_execution_time,
            priority_nodes: self.priority_nodes.clone(),
            token_budget: self.token_budget,
            tokens_used: self.tokens_used,
            requests_per_minute: self.requests_per_minute,
        }
    }

    fn reset_run_state(&mut self) {
        self.stop = false;
        self.tokens_used = 0;
        self.request_times.clear();
    }

    pub async fn execute(
        &mut self,
        inputs: NodeInputs,
    ) -> Result<ExecutionResults, SchedulerError> {
        self.reset_run_state();

        let dag = self.dag.clone();

        let mut execution_order = dag.get_topological_order()?;
        let critical_path = dag.compute_critical_path().unwrap_or_else(|_| {
            execution_order
                .iter()
                .map(|name| (name.clone(), 0.0))
                .collect()
        });

        if !self.priority_nodes.is_empty() {
            for node in self.priority_nodes.iter().rev() {
                if let Some(pos) = execution_order.iter().position(|n| n == node) {
                    let node_name = execution_order.remove(pos);
                    execution_order.insert(0, node_name);
                }
            }
        }

        let mut ready_nodes: Vec<String> = Vec::new();
        let mut waiting_nodes: HashMap<String, Vec<String>> = HashMap::new();

        for node_name in &execution_order {
            if let Some(node) = dag.get_node(node_name) {
                let deps = node.dependencies();
                if deps.is_empty() {
                    ready_nodes.push(node_name.clone());
                } else {
                    waiting_nodes.insert(node_name.clone(), deps);
                }
            }
        }

        let mut node_outputs: NodeOutputs = HashMap::new();
        let priority_set: HashSet<String> = self.priority_nodes.iter().cloned().collect();

        while !ready_nodes.is_empty() || !waiting_nodes.is_empty() {
            if self.stop {
                break;
            }

            ready_nodes
                .sort_by(|a, b| self.compare_nodes(a, b, &critical_path, &priority_set, &dag));

            let mut current_batch = Vec::new();
            for _ in 0..self.max_parallel_nodes {
                if let Some(node) = ready_nodes.first().cloned() {
                    ready_nodes.remove(0);
                    current_batch.push(node);
                } else {
                    break;
                }
            }

            if current_batch.is_empty() {
                if waiting_nodes.is_empty() {
                    break;
                }
                let executed: HashSet<String> = node_outputs.keys().cloned().collect();
                let mut progress = false;
                for node in waiting_nodes.keys().cloned().collect::<Vec<_>>() {
                    if let Some(deps) = waiting_nodes.get_mut(&node) {
                        deps.retain(|dep| !executed.contains(dep));
                        if deps.is_empty() {
                            ready_nodes.push(node.clone());
                            waiting_nodes.remove(&node);
                            progress = true;
                        }
                    }
                }
                if !progress {
                    return Err(SchedulerError::Deadlock(
                        waiting_nodes.keys().cloned().collect(),
                    ));
                }
                continue;
            }

            for node_name in current_batch {
                let maybe_result = self
                    .execute_single_node(&dag, &node_name, &inputs, &node_outputs)
                    .await?;

                if let Some(result) = maybe_result {
                    node_outputs.insert(node_name.clone(), result);
                }

                if self.stop {
                    break;
                }
            }

            if self.stop {
                break;
            }

            let executed: HashSet<String> = node_outputs.keys().cloned().collect();
            for node in waiting_nodes.keys().cloned().collect::<Vec<_>>() {
                if let Some(deps) = waiting_nodes.get_mut(&node) {
                    deps.retain(|dep| !executed.contains(dep));
                    if deps.is_empty() {
                        ready_nodes.push(node.clone());
                        waiting_nodes.remove(&node);
                    }
                }
            }
        }

        Ok(ExecutionResults {
            results: node_outputs,
        })
    }

    async fn execute_single_node(
        &mut self,
        dag: &Dag,
        node_name: &str,
        base_inputs: &NodeInputs,
        node_outputs: &NodeOutputs,
    ) -> Result<Option<NodeResult>, SchedulerError> {
        let node_ref = dag
            .get_node(node_name)
            .ok_or_else(|| SchedulerError::MissingNode(node_name.to_string()))?;

        let dependencies = node_ref.dependencies();
        let token_cost = node_ref.token_cost();
        let node_box = node_ref.clone_box();

        let mut node_inputs = base_inputs.clone();
        let mut dependency_inputs: HashMap<String, NodeResult> = HashMap::new();
        for dep in dependencies {
            if let Some(value) = node_outputs.get(&dep) {
                dependency_inputs.insert(dep.clone(), value.clone());
            }
        }

        self.enforce_token_budget(node_name, token_cost)?;
        self.enforce_rate_limit().await;
        self.apply_dependency_outputs(dag, node_name, &mut node_inputs, &dependency_inputs);

        if !self
            .run_hooks(HookStage::BeforeExecute, node_name, &node_inputs, None)
            .await?
        {
            return Ok(None);
        }

        if self.stop {
            return Ok(None);
        }

        let timeout_duration = self.max_execution_time;
        let start = Instant::now();
        let result = match timeout(timeout_duration, node_box.execute(&node_inputs)).await {
            Ok(Ok(value)) => value,
            Ok(Err(err)) => {
                return Err(SchedulerError::NodeExecution {
                    node: node_name.to_string(),
                    source: err,
                })
            }
            Err(_) => {
                return Err(SchedulerError::Timeout {
                    node: node_name.to_string(),
                    timeout_ms: timeout_duration.as_millis() as u64,
                })
            }
        };

        self.run_hooks(
            HookStage::AfterExecute,
            node_name,
            &node_inputs,
            Some(result.clone()),
        )
        .await?;

        let end = Instant::now();
        self.write_audit_entry(node_name, start, end, &node_inputs, &result);

        self.run_hooks(
            HookStage::Audit,
            node_name,
            &node_inputs,
            Some(result.clone()),
        )
        .await?;

        Ok(Some(result))
    }

    async fn run_hooks(
        &mut self,
        stage: HookStage,
        node_name: &str,
        inputs: &NodeInputs,
        output: Option<NodeResult>,
    ) -> Result<bool, SchedulerError> {
        if self.hooks.is_empty() {
            return Ok(true);
        }

        for hook in &self.hooks {
            let context = HookContext {
                stage: stage.clone(),
                node_name: node_name.to_string(),
                inputs: inputs.clone(),
                output: output.clone(),
                scheduler: self.snapshot(),
            };
            if hook.invoke(context).await == HookOutcome::Stop {
                self.stop = true;
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn enforce_token_budget(&mut self, node_name: &str, cost: u32) -> Result<(), SchedulerError> {
        if let Some(budget) = self.token_budget {
            if self.tokens_used + cost > budget {
                return Err(SchedulerError::TokenBudgetExceeded(node_name.to_string()));
            }
            self.tokens_used += cost;
        }
        Ok(())
    }

    async fn enforce_rate_limit(&mut self) {
        if let Some(limit) = self.requests_per_minute {
            if limit == 0 {
                return;
            }
            let limit = limit as usize;
            let now = Instant::now();
            while let Some(front) = self.request_times.front() {
                if now.duration_since(*front) >= Duration::from_secs(60) {
                    self.request_times.pop_front();
                } else {
                    break;
                }
            }
            if self.request_times.len() >= limit {
                if let Some(oldest) = self.request_times.front().copied() {
                    let elapsed = now.duration_since(oldest);
                    if elapsed < Duration::from_secs(60) {
                        let wait = Duration::from_secs(60) - elapsed;
                        tokio::time::sleep(wait).await;
                    }
                }
            }
            let now = Instant::now();
            while let Some(front) = self.request_times.front() {
                if now.duration_since(*front) >= Duration::from_secs(60) {
                    self.request_times.pop_front();
                } else {
                    break;
                }
            }
            self.request_times.push_back(now);
        }
    }

    fn apply_dependency_outputs(
        &self,
        dag: &Dag,
        node_name: &str,
        node_inputs: &mut NodeInputs,
        dependency_outputs: &HashMap<String, NodeResult>,
    ) {
        for (dep_name, dep_output) in dependency_outputs {
            let normalized_map = match dep_output {
                Value::Object(map) => map.clone(),
                other => {
                    let mut map = Map::new();
                    map.insert(dep_name.clone(), other.clone());
                    map
                }
            };

            if let Some(mapping) = dag
                .edge_mappings()
                .get(dep_name)
                .and_then(|targets| targets.get(node_name))
            {
                for (source_field, target_field) in mapping {
                    if let Some(value) = normalized_map.get(source_field) {
                        node_inputs.insert(target_field.clone(), value.clone());
                    }
                }
            } else {
                for (key, value) in &normalized_map {
                    node_inputs.insert(key.clone(), value.clone());
                }
            }

            node_inputs
                .entry(dep_name.clone())
                .or_insert_with(|| Value::Object(normalized_map.clone()));
        }
    }

    fn write_audit_entry(
        &self,
        node_name: &str,
        start: Instant,
        end: Instant,
        inputs: &NodeInputs,
        output: &NodeResult,
    ) {
        let Some(path) = &self.audit_file else {
            return;
        };

        let entry = json!({
            "node": node_name,
            "start": start.elapsed().as_secs_f64(),
            "end": end.elapsed().as_secs_f64(),
            "status": "success",
            "inputs": Value::Object(inputs.clone()),
            "output": output,
        });

        if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(path) {
            let _ = writeln!(file, "{}", entry);
        }
    }

    fn compare_nodes(
        &self,
        a: &str,
        b: &str,
        critical_path: &HashMap<String, f64>,
        priority_nodes: &HashSet<String>,
        dag: &Dag,
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        let key_a = self.priority_metrics(a, critical_path, priority_nodes, dag);
        let key_b = self.priority_metrics(b, critical_path, priority_nodes, dag);

        if key_a.is_critical != key_b.is_critical {
            return if key_a.is_critical {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }

        if key_a.is_redundancy != key_b.is_redundancy {
            return if !key_a.is_redundancy {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }

        match key_b
            .cp_score
            .partial_cmp(&key_a.cp_score)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => {}
            ord => return ord,
        }

        match key_a
            .level
            .partial_cmp(&key_b.level)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => {}
            ord => return ord,
        }

        a.cmp(b)
    }

    fn priority_metrics(
        &self,
        node_name: &str,
        critical_path: &HashMap<String, f64>,
        priority_nodes: &HashSet<String>,
        dag: &Dag,
    ) -> PriorityMetrics {
        let metadata = dag
            .get_node(node_name)
            .map(|node| node.metadata().clone())
            .unwrap_or_default();
        let tags = extract_tags(&metadata);
        let level = extract_level(&metadata);
        let is_redundancy = tags.contains(&"redundancy".to_string())
            || metadata
                .get("generated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let is_critical = priority_nodes.contains(node_name)
            || tags.contains(&"critical".to_string())
            || metadata
                .get("is_critical")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let cp_score = *critical_path.get(node_name).unwrap_or(&0.0);

        PriorityMetrics {
            is_critical,
            is_redundancy,
            cp_score,
            level,
        }
    }
}

#[derive(Debug)]
struct PriorityMetrics {
    is_critical: bool,
    is_redundancy: bool,
    cp_score: f64,
    level: f64,
}

fn extract_tags(metadata: &NodeMetadata) -> Vec<String> {
    match metadata.get("tags") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(Value::String(tag)) => vec![tag.clone()],
        _ => Vec::new(),
    }
}

fn extract_level(metadata: &NodeMetadata) -> f64 {
    match metadata.get("level") {
        Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0),
        Some(Value::String(text)) => text.parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(
        name: &str,
        func: impl Fn(&NodeInputs) -> NodeResult + Send + Sync + 'static,
    ) -> crate::nodes::ToolNode {
        crate::nodes::ToolNode::new_sync(name.to_string(), move |inputs| Ok(func(inputs)))
    }

    fn noop_inputs() -> NodeInputs {
        NodeInputs::new()
    }

    #[tokio::test]
    async fn scheduler_executes_simple_dag() {
        let mut dag = Dag::new("simple");
        dag.add_node(tool("start", |_| json!({"value": 1})));
        dag.add_node(tool("end", |inputs| {
            let value = inputs
                .get("start")
                .and_then(Value::as_object)
                .and_then(|m| m.get("value"))
                .and_then(Value::as_i64)
                .unwrap_or_default();
            json!({"total": value + 1})
        }));
        dag.add_edge("start", "end", None).unwrap();

        let mut scheduler = Scheduler::new(dag);
        let results = scheduler.execute(noop_inputs()).await.unwrap();
        assert_eq!(results.results["start"]["value"], 1);
        assert_eq!(results.results["end"]["total"], 2);
    }

    #[tokio::test]
    async fn scheduler_enforces_token_budget() {
        let mut dag = Dag::new("budget");
        dag.add_node(crate::nodes::ToolNode::new_sync("a", |_| Ok(json!(null))).with_token_cost(5));
        dag.add_node(crate::nodes::ToolNode::new_sync("b", |_| Ok(json!(null))).with_token_cost(2));
        dag.add_edge("a", "b", None).unwrap();

        let mut scheduler = Scheduler::new(dag);
        scheduler.set_token_budget(Some(6));

        let err = scheduler.execute(noop_inputs()).await.unwrap_err();
        matches!(err, SchedulerError::TokenBudgetExceeded(node) if node == "b");
    }

    #[tokio::test]
    async fn scheduler_hooks_can_stop_execution() {
        let mut dag = Dag::new("hooks");
        dag.add_node(tool("alpha", |_| json!({"done": true})));
        dag.add_node(tool("beta", |_| json!({"done": true})));
        dag.add_edge("alpha", "beta", None).unwrap();

        let mut scheduler = Scheduler::new(dag);
        let hook = std::sync::Arc::new(|ctx: HookContext| async move {
            if ctx.stage == HookStage::BeforeExecute && ctx.node_name == "beta" {
                HookOutcome::Stop
            } else {
                HookOutcome::Continue
            }
        });
        scheduler.set_hooks(vec![hook]);

        let results = scheduler.execute(noop_inputs()).await.unwrap();
        assert!(results.results.contains_key("alpha"));
        assert!(!results.results.contains_key("beta"));
    }
}
