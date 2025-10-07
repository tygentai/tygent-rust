#![allow(dead_code)]

use std::sync::Arc;

use async_openai::types::CreateCompletionRequestArgs;
use async_trait::async_trait;
use serde_json::Value;

use crate::dag::{Dag, NodeOutputs};
use crate::nodes::{NodeExecutionError, NodeInputs, ToolNode};
use crate::scheduler::{Scheduler, SchedulerError};
use crate::service_bridge::{default_registry, LLMRuntimeError};

#[derive(Clone, Debug)]
pub struct OpenAIRequest {
    pub model: String,
    pub max_tokens: Option<u16>,
    pub temperature: Option<f32>,
}

impl Default for OpenAIRequest {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo-instruct".into(),
            max_tokens: Some(256),
            temperature: Some(0.7),
        }
    }
}

#[async_trait]
pub trait OpenAIExecutor: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        request: &OpenAIRequest,
    ) -> Result<String, NodeExecutionError>;
}

pub struct AsyncOpenAIExecutor {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
}

impl AsyncOpenAIExecutor {
    pub fn new() -> Self {
        Self {
            client: async_openai::Client::new(),
        }
    }
}

#[async_trait]
impl OpenAIExecutor for AsyncOpenAIExecutor {
    async fn generate(
        &self,
        prompt: String,
        request: &OpenAIRequest,
    ) -> Result<String, NodeExecutionError> {
        let request = CreateCompletionRequestArgs::default()
            .model(request.model.clone())
            .prompt(prompt)
            .max_tokens(request.max_tokens.unwrap_or(256))
            .temperature(request.temperature.unwrap_or(0.7))
            .build()
            .map_err(|err| NodeExecutionError::execution("openai", err.to_string()))?;
        let response = self
            .client
            .completions()
            .create(request)
            .await
            .map_err(|err| NodeExecutionError::execution("openai", err.to_string()))?;
        let text = response
            .choices
            .first()
            .map(|choice| choice.text.clone())
            .unwrap_or_default();
        if text.is_empty() {
            Err(NodeExecutionError::execution("openai", "empty response"))
        } else {
            Ok(text)
        }
    }
}

#[derive(Clone)]
pub struct OpenAICodexNode {
    name: String,
    prompt_template: String,
    dependencies: Vec<String>,
    request: OpenAIRequest,
}

impl OpenAICodexNode {
    pub fn new(
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        dependencies: Vec<String>,
        request: OpenAIRequest,
    ) -> Self {
        Self {
            name: name.into(),
            prompt_template: prompt_template.into(),
            dependencies,
            request,
        }
    }

    pub fn into_tool_node(self, executor: Arc<dyn OpenAIExecutor>) -> ToolNode {
        let prompt_template = self.prompt_template.clone();
        let dependencies = self.dependencies.clone();
        let request = self.request.clone();
        ToolNode::new(self.name.clone(), move |inputs: &NodeInputs| {
            let prompt = render_template(&prompt_template, inputs);
            let executor = executor.clone();
            let request = request.clone();
            async move {
                let content = executor.generate(prompt, &request).await?;
                Ok(Value::String(content))
            }
        })
        .with_dependencies(dependencies)
    }
}

pub struct OpenAICodexIntegration {
    dag: Dag,
    executor: Arc<dyn OpenAIExecutor>,
    max_parallel: Option<usize>,
    max_execution_ms: Option<u64>,
}

impl OpenAICodexIntegration {
    pub fn new(executor: Arc<dyn OpenAIExecutor>) -> Self {
        Self {
            dag: Dag::new("openai_codex_dag"),
            executor,
            max_parallel: None,
            max_execution_ms: None,
        }
    }

    pub fn with_default_client() -> Self {
        Self::new(Arc::new(AsyncOpenAIExecutor::new()))
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        dependencies: Vec<String>,
        request: OpenAIRequest,
    ) -> ToolNode {
        let node = OpenAICodexNode::new(name, prompt_template, dependencies.clone(), request)
            .into_tool_node(self.executor.clone());
        self.dag.add_node(node.clone());
        node
    }

    pub fn optimize(&mut self, max_parallel: Option<usize>, max_execution_ms: Option<u64>) {
        self.max_parallel = max_parallel;
        self.max_execution_ms = max_execution_ms;
    }

    pub async fn execute(&self, inputs: NodeInputs) -> Result<NodeOutputs, SchedulerError> {
        let mut scheduler = Scheduler::new(self.dag.clone());
        if let Some(value) = self.max_parallel {
            scheduler.set_max_parallel_nodes(value);
        }
        if let Some(value) = self.max_execution_ms {
            scheduler.set_max_execution_time(value);
        }
        scheduler.execute(inputs).await.map(|res| res.results)
    }
}

pub fn patch() -> crate::patch::PatchResult {
    let registry = default_registry();
    let executor = Arc::new(AsyncOpenAIExecutor::new());

    let handler_executor = executor.clone();
    registry.register(
        "openai",
        Arc::new(move |prompt: String, metadata: Value, _inputs: Value| {
            let executor = handler_executor.clone();
            Box::pin(async move {
                let request = request_from_metadata(&metadata)?;
                executor
                    .generate(prompt, &request)
                    .await
                    .map(Value::String)
                    .map_err(|err| LLMRuntimeError::Handler(err.to_string()))
            })
        }),
    );

    Ok(())
}

fn render_template(template: &str, inputs: &NodeInputs) -> String {
    let mut rendered = template.to_string();
    for (key, value) in inputs {
        if let Some(text) = value.as_str() {
            rendered = rendered.replace(&format!("{{{}}}", key), text);
        }
    }
    rendered
}

fn request_from_metadata(metadata: &Value) -> Result<OpenAIRequest, LLMRuntimeError> {
    let mut request = OpenAIRequest::default();
    if let Some(obj) = metadata.as_object() {
        if let Some(model) = obj.get("model").and_then(Value::as_str) {
            request.model = model.to_string();
        }
        if let Some(tokens) = obj
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u16)
        {
            request.max_tokens = Some(tokens);
        }
        if let Some(temp) = obj.get("temperature").and_then(Value::as_f64) {
            request.temperature = Some(temp as f32);
        }
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    struct MockExecutor;

    #[async_trait]
    impl OpenAIExecutor for MockExecutor {
        async fn generate(
            &self,
            prompt: String,
            _request: &OpenAIRequest,
        ) -> Result<String, NodeExecutionError> {
            sleep(Duration::from_millis(5)).await;
            Ok(prompt.to_uppercase())
        }
    }

    #[tokio::test]
    async fn integration_runs_prompt() {
        let executor: Arc<dyn OpenAIExecutor> = Arc::new(MockExecutor);
        let mut integration = OpenAICodexIntegration::new(executor);
        integration.add_node("codex", "print {code}", vec![], OpenAIRequest::default());
        let mut inputs = NodeInputs::new();
        inputs.insert("code".into(), Value::String("x".into()));
        let results = integration.execute(inputs).await.unwrap();
        assert_eq!(
            results.get("codex").unwrap(),
            &Value::String("PRINT X".into())
        );
    }
}
