use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

use crate::dag::{Dag, NodeOutputs};
use crate::nodes::{NodeExecutionError, NodeInputs, ToolNode};
use crate::scheduler::{Scheduler, SchedulerError};
use crate::service_bridge::{default_registry, LLMRuntimeError};

#[derive(Clone, Debug)]
pub struct GoogleAIRequest {
    pub model: String,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for GoogleAIRequest {
    fn default() -> Self {
        Self {
            model: "gemini-1.5-pro".into(),
            max_output_tokens: Some(256),
            temperature: Some(0.7),
        }
    }
}

#[async_trait]
pub trait GoogleAIExecutor: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        request: &GoogleAIRequest,
    ) -> Result<String, NodeExecutionError>;
}

pub struct HttpGoogleAIExecutor {
    client: Client,
    api_key: String,
}

impl HttpGoogleAIExecutor {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
        }
    }

    pub fn from_env() -> Result<Self, NodeExecutionError> {
        let key = std::env::var("GOOGLE_API_KEY")
            .map_err(|_| NodeExecutionError::execution("google_ai", "GOOGLE_API_KEY missing"))?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl GoogleAIExecutor for HttpGoogleAIExecutor {
    async fn generate(
        &self,
        prompt: String,
        request: &GoogleAIRequest,
    ) -> Result<String, NodeExecutionError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            request.model, self.api_key
        );

        let mut body = serde_json::json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        });
        if request.max_output_tokens.is_some() || request.temperature.is_some() {
            body["generationConfig"] = serde_json::json!({
                "maxOutputTokens": request.max_output_tokens,
                "temperature": request.temperature,
            });
        }

        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|err| NodeExecutionError::execution("google_ai", err.to_string()))?
            .error_for_status()
            .map_err(|err| NodeExecutionError::execution("google_ai", err.to_string()))?
            .json::<Value>()
            .await
            .map_err(|err| NodeExecutionError::execution("google_ai", err.to_string()))?;

        let text = response
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .ok_or_else(|| NodeExecutionError::execution("google_ai", "empty response"))?;

        Ok(text)
    }
}

#[derive(Clone)]
pub struct GoogleAINode {
    name: String,
    prompt_template: String,
    dependencies: Vec<String>,
    request: GoogleAIRequest,
}

impl GoogleAINode {
    pub fn new(
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        dependencies: Vec<String>,
        request: GoogleAIRequest,
    ) -> Self {
        Self {
            name: name.into(),
            prompt_template: prompt_template.into(),
            dependencies,
            request,
        }
    }

    pub fn into_tool_node(self, executor: Arc<dyn GoogleAIExecutor>) -> ToolNode {
        let prompt_template = self.prompt_template.clone();
        let dependencies = self.dependencies.clone();
        let request = self.request.clone();
        ToolNode::new(self.name.clone(), move |inputs: &NodeInputs| {
            let prompt = render_template(&prompt_template, inputs);
            let executor = executor.clone();
            let request = request.clone();
            async move {
                let text = executor.generate(prompt, &request).await?;
                Ok(Value::String(text))
            }
        })
        .with_dependencies(dependencies)
    }
}

pub struct GoogleAIIntegration {
    dag: Dag,
    executor: Arc<dyn GoogleAIExecutor>,
    max_parallel: Option<usize>,
    max_execution_ms: Option<u64>,
}

impl GoogleAIIntegration {
    pub fn new(executor: Arc<dyn GoogleAIExecutor>) -> Self {
        Self {
            dag: Dag::new("google_ai_dag"),
            executor,
            max_parallel: None,
            max_execution_ms: None,
        }
    }

    pub fn with_default_client() -> Result<Self, NodeExecutionError> {
        let executor = HttpGoogleAIExecutor::from_env()?;
        Ok(Self::new(Arc::new(executor)))
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        dependencies: Vec<String>,
        request: GoogleAIRequest,
    ) -> ToolNode {
        let node = GoogleAINode::new(name, prompt_template, dependencies.clone(), request)
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
    let executor = HttpGoogleAIExecutor::from_env()
        .map(|exec| Arc::new(exec) as Arc<dyn GoogleAIExecutor>)
        .map_err(|err| err.to_string())?;

    let handler_executor = executor.clone();
    registry.register(
        "google_ai",
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

fn request_from_metadata(metadata: &Value) -> Result<GoogleAIRequest, LLMRuntimeError> {
    let mut request = GoogleAIRequest::default();
    if let Some(obj) = metadata.as_object() {
        if let Some(model) = obj.get("model").and_then(Value::as_str) {
            request.model = model.to_string();
        }
        if let Some(tokens) = obj
            .get("max_output_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
        {
            request.max_output_tokens = Some(tokens);
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
    impl GoogleAIExecutor for MockExecutor {
        async fn generate(
            &self,
            prompt: String,
            _request: &GoogleAIRequest,
        ) -> Result<String, NodeExecutionError> {
            sleep(Duration::from_millis(5)).await;
            Ok(prompt.to_uppercase())
        }
    }

    #[tokio::test]
    async fn integration_executes_nodes() {
        let executor: Arc<dyn GoogleAIExecutor> = Arc::new(MockExecutor);
        let mut manager = GoogleAIIntegration::new(executor);
        manager.add_node("draft", "Hello {user}", vec![], GoogleAIRequest::default());
        let mut inputs = NodeInputs::new();
        inputs.insert("user".into(), Value::String("gemini".into()));

        let results = manager.execute(inputs).await.unwrap();
        let draft = results.get("draft").unwrap();
        assert_eq!(draft, &Value::String("HELLO GEMINI".into()));
    }
}
