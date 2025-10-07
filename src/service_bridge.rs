use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;
use serde_json::{json, Map, Value};
use tokio::time::{sleep, Duration};

use crate::nodes::{NodeExecutionError, NodeInputs, NodeMetadata, ToolNode};
use crate::plan_parser::{parse_plan, ParsePlanError, PlanSpec, PlanStep};
use crate::prefetch::prefetch_many;
use crate::scheduler::{Scheduler, SchedulerError};

pub type NodeOutputs = crate::dag::NodeOutputs;

pub type PromptFuture = Pin<Box<dyn Future<Output = Result<Value, LLMRuntimeError>> + Send>>;

pub trait PromptHandler: Send + Sync {
    fn invoke(&self, prompt: String, metadata: Value, inputs: Value) -> PromptFuture;
}

impl<F, Fut> PromptHandler for F
where
    F: Send + Sync + 'static + Fn(String, Value, Value) -> Fut,
    Fut: Future<Output = Result<Value, LLMRuntimeError>> + Send + 'static,
{
    fn invoke(&self, prompt: String, metadata: Value, inputs: Value) -> PromptFuture {
        Box::pin((self)(prompt, metadata, inputs))
    }
}

#[derive(Default)]
pub struct LLMRuntimeRegistry {
    handlers: RwLock<HashMap<String, Arc<dyn PromptHandler>>>,
}

impl LLMRuntimeRegistry {
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, provider: impl Into<String>, handler: Arc<dyn PromptHandler>) {
        let mut handlers = self.handlers.write().expect("LLMRuntimeRegistry poisoned");
        handlers.insert(provider.into(), handler);
    }

    pub async fn call(
        &self,
        provider: &str,
        prompt: String,
        metadata: Value,
        inputs: Value,
    ) -> Result<Value, LLMRuntimeError> {
        let handler = {
            let handlers = self.handlers.read().expect("LLMRuntimeRegistry poisoned");
            handlers
                .get(provider)
                .cloned()
                .ok_or_else(|| LLMRuntimeError::ProviderNotFound(provider.to_string()))?
        };

        handler.invoke(prompt, metadata, inputs).await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LLMRuntimeError {
    #[error("no runtime registered for provider '{0}'")]
    ProviderNotFound(String),
    #[error("handler error: {0}")]
    Handler(String),
}

static DEFAULT_REGISTRY: Lazy<Arc<LLMRuntimeRegistry>> = Lazy::new(|| {
    let registry = Arc::new(LLMRuntimeRegistry::new());
    let echo = Arc::new(
        |prompt: String, metadata: Value, inputs: Value| async move {
            let mut payload = Map::new();
            payload.insert("prompt".to_string(), Value::String(prompt));
            payload.insert("metadata".to_string(), metadata);
            payload.insert("inputs".to_string(), inputs);
            Ok(Value::Object(payload))
        },
    );
    registry.register("echo", echo);
    registry
});

pub fn default_registry() -> Arc<LLMRuntimeRegistry> {
    DEFAULT_REGISTRY.clone()
}

#[derive(Clone)]
pub struct ServicePlan {
    pub plan: PlanSpec,
    pub prefetch_links: Vec<String>,
    pub raw: Value,
}

pub struct ServicePlanBuilder {
    registry: Arc<LLMRuntimeRegistry>,
}

impl ServicePlanBuilder {
    pub fn new(registry: Option<Arc<LLMRuntimeRegistry>>) -> Self {
        Self {
            registry: registry.unwrap_or_else(default_registry),
        }
    }

    pub fn build(&self, payload: &Value) -> Result<ServicePlan, ServicePlanError> {
        let steps_value = payload
            .get("steps")
            .and_then(Value::as_array)
            .ok_or(ServicePlanError::MissingSteps)?;

        let mut steps: Vec<PlanStep> = Vec::with_capacity(steps_value.len());

        for step_value in steps_value {
            let step = StepConfig::from_value(step_value)?;
            let registry = self.registry.clone();
            let name = step.name.clone();
            let prompt = step.prompt.clone();
            let kind = step.kind.clone();
            let links = step.links.clone();
            let dependencies = step.dependencies.clone();
            let metadata_map = step.metadata.clone();
            let metadata_for_node = metadata_map.clone();
            let metadata_for_payload = metadata_map.clone();
            let metadata_value = Value::Object(metadata_map.clone());
            let provider = metadata_map
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("echo")
                .to_string();
            let delay_duration = simulate_delay(&metadata_map);
            let is_critical = step.is_critical;
            let latency_estimate = step.latency_estimate;
            let token_cost = step.token_cost;

            let mut node = ToolNode::new(name.clone(), move |inputs: &NodeInputs| {
                let registry = registry.clone();
                let prompt_template = prompt.clone();
                let kind = kind.clone();
                let links = links.clone();
                let metadata_for_payload = metadata_for_payload.clone();
                let metadata_value = metadata_value.clone();
                let provider = provider.clone();
                let name = name.clone();
                let rendered_prompt = render_prompt(&prompt_template, inputs);
                let inputs_value = Value::Object(inputs.clone());
                async move {
                    if let Some(delay) = delay_duration {
                        sleep(delay).await;
                    }

                    let mut payload = Map::new();
                    payload.insert("step".into(), Value::String(name.clone()));
                    payload.insert("prompt".into(), Value::String(rendered_prompt.clone()));
                    payload.insert("inputs".into(), inputs_value.clone());
                    payload.insert(
                        "links".into(),
                        Value::Array(links.iter().cloned().map(Value::String).collect::<Vec<_>>()),
                    );
                    payload.insert("kind".into(), Value::String(kind.clone()));
                    payload.insert(
                        "metadata".into(),
                        Value::Object(metadata_for_payload.clone()),
                    );

                    let result = if kind == "llm" {
                        registry
                            .call(
                                &provider,
                                rendered_prompt.clone(),
                                metadata_value.clone(),
                                inputs_value,
                            )
                            .await
                            .map_err(|err| {
                                NodeExecutionError::execution(name.clone(), err.to_string())
                            })?
                    } else {
                        json!({ "echo": rendered_prompt })
                    };
                    payload.insert("result".into(), result);
                    Ok(Value::Object(payload))
                }
            })
            .with_dependencies(dependencies.clone())
            .with_metadata(metadata_for_node.clone());

            if let Some(cost) = token_cost {
                node = node.with_token_cost(cost);
            }
            if let Some(latency) = latency_estimate {
                node = node.with_latency_estimate(latency);
            }

            let mut plan_step = PlanStep::new(node).with_dependencies(dependencies);
            if is_critical {
                plan_step = plan_step.critical(true);
            }
            steps.push(plan_step);
        }

        let plan_spec = PlanSpec::new(
            payload
                .get("name")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            steps,
        );

        let prefetch_links = payload
            .get("prefetch")
            .and_then(Value::as_object)
            .and_then(|obj| obj.get("links"))
            .and_then(Value::as_array)
            .map(|arr| {
                let mut seen = HashSet::new();
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|url| seen.insert(url.to_string()))
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(ServicePlan {
            plan: plan_spec,
            prefetch_links,
            raw: payload.clone(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServicePlanError {
    #[error("service payload missing 'steps' list")]
    MissingSteps,
    #[error("step missing required field '{0}'")]
    MissingField(String),
    #[error("invalid field '{0}'")]
    InvalidField(String),
}

fn simulate_delay(metadata: &NodeMetadata) -> Option<Duration> {
    let keys = [
        "simulated_duration",
        "latency_estimate",
        "estimated_duration",
    ];
    let mut candidate: Option<&Value> = None;
    for key in keys {
        if let Some(value) = metadata.get(key) {
            candidate = Some(value);
            break;
        }
    }

    match candidate {
        Some(Value::Number(number)) => number.as_f64().map(|f| Duration::from_secs_f64(f.max(0.0))),
        Some(Value::String(text)) => text
            .parse::<f64>()
            .ok()
            .map(|f| Duration::from_secs_f64(f.max(0.0))),
        Some(Value::Bool(flag)) => {
            if *flag {
                Some(Duration::from_millis(10))
            } else {
                None
            }
        }
        Some(_) => None,
        None => Some(Duration::from_millis(10)),
    }
}

struct StepConfig {
    name: String,
    prompt: String,
    kind: String,
    dependencies: Vec<String>,
    metadata: NodeMetadata,
    links: Vec<String>,
    is_critical: bool,
    token_cost: Option<u32>,
    latency_estimate: Option<f64>,
}

impl StepConfig {
    fn from_value(value: &Value) -> Result<Self, ServicePlanError> {
        let obj = value
            .as_object()
            .ok_or_else(|| ServicePlanError::InvalidField("step".into()))?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| ServicePlanError::MissingField("name".into()))?
            .to_string();
        let prompt = obj
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let kind = obj
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("llm")
            .to_string();
        let dependencies = obj
            .get("dependencies")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let metadata = obj
            .get("metadata")
            .and_then(Value::as_object)
            .map(|m| m.clone())
            .unwrap_or_else(Map::new);
        let tags = obj
            .get("tags")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut metadata = metadata;
        merge_tags(&mut metadata, &tags);
        if let Some(level) = obj.get("level") {
            if !metadata.contains_key("level") {
                metadata.insert("level".into(), level.clone());
            }
        }
        if !metadata.contains_key("links") {
            metadata.insert(
                "links".into(),
                Value::Array(
                    obj.get("links")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                ),
            );
        }
        if !metadata.contains_key("prompt") {
            metadata.insert("prompt".into(), Value::String(prompt.clone()));
        }
        metadata.insert("kind".into(), Value::String(kind.clone()));
        if let Some(is_critical) = obj.get("is_critical").and_then(Value::as_bool) {
            metadata.insert("is_critical".into(), Value::Bool(is_critical));
        }

        let links = obj
            .get("links")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let is_critical = obj
            .get("is_critical")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let token_cost = obj
            .get("metadata")
            .and_then(|m| m.get("token_estimate"))
            .or_else(|| metadata.get("token_estimate"))
            .and_then(Value::as_i64)
            .map(|v| v.max(0) as u32);
        let latency_estimate = obj
            .get("latency_estimate")
            .and_then(Value::as_f64)
            .or_else(|| metadata.get("latency_estimate").and_then(Value::as_f64));

        Ok(Self {
            name,
            prompt,
            kind,
            dependencies,
            metadata,
            links,
            is_critical,
            token_cost,
            latency_estimate,
        })
    }
}

fn merge_tags(metadata: &mut NodeMetadata, tags: &[String]) {
    let mut existing = metadata
        .get("tags")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut seen: HashSet<String> = existing
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    for tag in tags {
        if seen.insert(tag.clone()) {
            existing.push(Value::String(tag.clone()));
        }
    }
    if !existing.is_empty() {
        metadata.insert("tags".into(), Value::Array(existing));
    }
}

pub async fn execute_service_plan(
    service_plan: ServicePlan,
    mut inputs: NodeInputs,
    max_parallel_nodes: Option<usize>,
) -> Result<NodeOutputs, ServiceExecutionError> {
    let mut prefetch_results = None;
    if !service_plan.prefetch_links.is_empty() {
        let fetched = prefetch_many(service_plan.prefetch_links.clone()).await;
        let mut map = Map::new();
        for (k, v) in fetched {
            map.insert(k, Value::String(v));
        }
        prefetch_results = Some(Value::Object(map));
    }

    if let Some(prefetch) = prefetch_results {
        inputs.insert("prefetch".into(), prefetch);
    }

    let (dag, critical) = parse_plan(service_plan.plan).map_err(ServiceExecutionError::Parse)?;
    let mut scheduler = Scheduler::new(dag);
    if let Some(limit) = max_parallel_nodes {
        scheduler.set_max_parallel_nodes(limit);
    }
    scheduler.set_priority_nodes(critical);
    let results = scheduler
        .execute(inputs)
        .await
        .map_err(ServiceExecutionError::Scheduler)?;
    Ok(results.results)
}

#[derive(Debug, thiserror::Error)]
pub enum ServiceExecutionError {
    #[error("plan parsing failed: {0}")]
    Parse(ParsePlanError),
    #[error("scheduler execution failed: {0}")]
    Scheduler(SchedulerError),
}

fn render_prompt(template: &str, inputs: &NodeInputs) -> String {
    if template.is_empty() {
        return String::new();
    }

    let mut result = String::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            let mut token = String::new();
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == '}' {
                    break;
                }
                token.push(next);
            }

            if !token.is_empty() {
                if let Some(value) = lookup(inputs, token.trim()).and_then(value_to_string) {
                    result.push_str(&value);
                    continue;
                }
            }

            result.push('{');
            result.push_str(&token);
            result.push('}');
        } else {
            result.push(ch);
        }
    }

    result
}

fn lookup(inputs: &NodeInputs, path: &str) -> Option<Value> {
    let mut current: Value = Value::Object(inputs.clone());
    let mut token = String::new();
    let mut chars = path.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '.' => {
                if !token.is_empty() {
                    current = advance(current, &token)?;
                    token.clear();
                }
            }
            '[' => {
                if !token.is_empty() {
                    current = advance(current, &token)?;
                    token.clear();
                }
                let mut index = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == ']' {
                        break;
                    }
                    index.push(c);
                }
                current = advance(current, &index)?;
            }
            _ => token.push(ch),
        }
    }

    if !token.is_empty() {
        current = advance(current, &token)?;
    }

    Some(current)
}

fn advance(value: Value, key: &str) -> Option<Value> {
    match value {
        Value::Object(map) => map.get(key).cloned(),
        Value::Array(arr) => {
            if let Ok(idx) = key.parse::<usize>() {
                arr.get(idx).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

fn value_to_string(value: Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn service_plan_executes_with_default_registry() {
        let payload = json!({
            "name": "demo",
            "steps": [
                {
                    "name": "draft",
                    "prompt": "Hello {user.name}",
                    "metadata": {
                        "provider": "echo"
                    }
                }
            ],
            "prefetch": {
                "links": ["https://example.com/a"]
            }
        });

        let builder = ServicePlanBuilder::new(None);
        let plan = builder.build(&payload).unwrap();
        let mut inputs = NodeInputs::new();
        let mut user = Map::new();
        user.insert("name".into(), Value::String("Alice".into()));
        inputs.insert("user".into(), Value::Object(user));

        let result = execute_service_plan(plan, inputs, Some(1)).await.unwrap();
        assert!(result.contains_key("draft"));
        let payload = result.get("draft").unwrap();
        let prompt = payload.get("prompt").and_then(Value::as_str).unwrap();
        assert_eq!(prompt, "Hello Alice");
        assert!(payload.get("result").is_some());
        let inputs = payload.get("inputs").and_then(Value::as_object).unwrap();
        assert!(inputs.contains_key("prefetch"));
    }
}
