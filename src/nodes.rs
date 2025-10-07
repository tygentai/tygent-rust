use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};
use thiserror::Error;

pub type NodeInputs = Map<String, Value>;
pub type NodeResult = Value;
pub type NodeMetadata = Map<String, Value>;

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<NodeResult, NodeExecutionError>> + Send>>;

type ToolFunc = dyn Fn(&NodeInputs) -> ToolFuture + Send + Sync;
type ToolLatencyModel = dyn Fn(&ToolNode) -> f64 + Send + Sync;
type LLMLatencyModel = dyn Fn(&LLMNode) -> f64 + Send + Sync;

#[derive(Debug, Error)]
pub enum NodeExecutionError {
    #[error("node `{node}` execution failed: {message}")]
    Execution { node: String, message: String },
    #[error("execute not implemented for node `{node}`")]
    NotImplemented { node: String },
}

impl NodeExecutionError {
    pub fn execution(node: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Execution {
            node: node.into(),
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait Node: Send + Sync {
    fn name(&self) -> &str;
    fn dependencies(&self) -> Vec<String>;
    fn set_dependencies(&mut self, dependencies: Vec<String>);
    fn token_cost(&self) -> u32;
    fn latency_estimate(&self) -> f64;
    fn metadata(&self) -> &NodeMetadata;
    fn metadata_mut(&mut self) -> &mut NodeMetadata;
    fn clone_box(&self) -> Box<dyn Node>;

    async fn execute(&self, inputs: &NodeInputs) -> Result<NodeResult, NodeExecutionError>;
}

impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone)]
pub struct ToolNode {
    name: String,
    dependencies: Vec<String>,
    token_cost: u32,
    latency_estimate: Option<f64>,
    latency_model: Arc<ToolLatencyModel>,
    metadata: NodeMetadata,
    func: Arc<ToolFunc>,
}

impl ToolNode {
    pub fn new<F, Fut>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&NodeInputs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeResult, NodeExecutionError>> + Send + 'static,
    {
        let func_arc: Arc<ToolFunc> = Arc::new(move |inputs: &NodeInputs| {
            let fut = func(inputs);
            Box::pin(fut) as ToolFuture
        });

        Self {
            name: name.into(),
            dependencies: Vec::new(),
            token_cost: 0,
            latency_estimate: None,
            latency_model: Arc::new(default_tool_latency_model),
            metadata: NodeMetadata::new(),
            func: func_arc,
        }
    }

    pub fn new_sync<F>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&NodeInputs) -> Result<NodeResult, NodeExecutionError> + Send + Sync + 'static,
    {
        Self::new(name, move |inputs: &NodeInputs| {
            let result = func(inputs);
            async move { result }
        })
    }

    pub fn with_token_cost(mut self, token_cost: u32) -> Self {
        self.token_cost = token_cost;
        self
    }

    pub fn with_latency_estimate(mut self, latency: f64) -> Self {
        self.latency_estimate = Some(latency);
        self
    }

    pub fn with_latency_model<F>(mut self, model: F) -> Self
    where
        F: Fn(&ToolNode) -> f64 + Send + Sync + 'static,
    {
        self.latency_model = Arc::new(model);
        self
    }

    pub fn with_metadata(mut self, metadata: NodeMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }
}

#[async_trait]
impl Node for ToolNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn dependencies(&self) -> Vec<String> {
        self.dependencies.clone()
    }

    fn set_dependencies(&mut self, dependencies: Vec<String>) {
        self.dependencies = dependencies;
    }

    fn token_cost(&self) -> u32 {
        self.token_cost
    }

    fn latency_estimate(&self) -> f64 {
        if let Some(latency) = self.latency_estimate {
            latency
        } else {
            (self.latency_model)(self)
        }
    }

    fn metadata(&self) -> &NodeMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut NodeMetadata {
        &mut self.metadata
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    async fn execute(&self, inputs: &NodeInputs) -> Result<NodeResult, NodeExecutionError> {
        (self.func)(inputs).await
    }
}

#[derive(Clone)]
pub struct LLMNode {
    name: String,
    dependencies: Vec<String>,
    token_cost: u32,
    latency_estimate: Option<f64>,
    latency_model: Arc<LLMLatencyModel>,
    metadata: NodeMetadata,
    pub model: Option<String>,
    pub prompt_template: String,
}

impl LLMNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dependencies: Vec::new(),
            token_cost: 0,
            latency_estimate: None,
            latency_model: Arc::new(default_llm_latency_model),
            metadata: NodeMetadata::new(),
            model: None,
            prompt_template: String::new(),
        }
    }

    pub fn with_token_cost(mut self, token_cost: u32) -> Self {
        self.token_cost = token_cost;
        self
    }

    pub fn with_latency_estimate(mut self, latency: f64) -> Self {
        self.latency_estimate = Some(latency);
        self
    }

    pub fn with_latency_model<F>(mut self, model: F) -> Self
    where
        F: Fn(&LLMNode) -> f64 + Send + Sync + 'static,
    {
        self.latency_model = Arc::new(model);
        self
    }

    pub fn with_metadata(mut self, metadata: NodeMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }
}

#[async_trait]
impl Node for LLMNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn dependencies(&self) -> Vec<String> {
        self.dependencies.clone()
    }

    fn set_dependencies(&mut self, dependencies: Vec<String>) {
        self.dependencies = dependencies;
    }

    fn token_cost(&self) -> u32 {
        self.token_cost
    }

    fn latency_estimate(&self) -> f64 {
        if let Some(latency) = self.latency_estimate {
            latency
        } else {
            (self.latency_model)(self)
        }
    }

    fn metadata(&self) -> &NodeMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut NodeMetadata {
        &mut self.metadata
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    async fn execute(&self, _inputs: &NodeInputs) -> Result<NodeResult, NodeExecutionError> {
        Err(NodeExecutionError::NotImplemented {
            node: self.name.clone(),
        })
    }
}

pub fn default_llm_latency_model(node: &LLMNode) -> f64 {
    0.5 + 0.01 * node.token_cost as f64
}

pub fn default_tool_latency_model(_node: &ToolNode) -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn tool_node_executes_async_function() {
        let node = ToolNode::new("async_tool", |_inputs: &NodeInputs| async move {
            Ok(json!({"status": "ok"}))
        });

        let inputs = NodeInputs::new();
        let output = node.execute(&inputs).await.unwrap();
        assert_eq!(output["status"], "ok");
    }

    #[tokio::test]
    async fn tool_node_executes_sync_function() {
        let node = ToolNode::new_sync("sync_tool", |_inputs: &NodeInputs| Ok(json!({"value": 42})));

        let inputs = NodeInputs::new();
        let output = node.execute(&inputs).await.unwrap();
        assert_eq!(output["value"], 42);
    }

    #[test]
    fn default_latency_models_work() {
        let tool = ToolNode::new_sync("tool", |_inputs| Ok(json!(null)));
        assert!((tool.latency_estimate() - 0.1).abs() < f64::EPSILON);

        let llm = LLMNode::new("llm").with_token_cost(100);
        assert!((llm.latency_estimate() - 1.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn llm_node_execute_not_implemented() {
        let llm = LLMNode::new("llm-node");
        let inputs = NodeInputs::new();
        let err = llm.execute(&inputs).await.unwrap_err();
        match err {
            NodeExecutionError::NotImplemented { node } => assert_eq!(node, "llm-node"),
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn node_clone_box_yields_distinct_instances() {
        let tool = ToolNode::new_sync("clone", |_inputs| Ok(json!(null)))
            .with_dependencies(vec!["a".into()])
            .with_token_cost(5);
        let boxed: Box<dyn Node> = Box::new(tool);
        let cloned = boxed.clone();
        assert_eq!(boxed.name(), cloned.name());
        assert_eq!(boxed.dependencies(), cloned.dependencies());
        assert!((boxed.latency_estimate() - cloned.latency_estimate()).abs() < f64::EPSILON);
    }
}
