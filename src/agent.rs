use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::nodes::NodeInputs;

pub type AgentInputs = NodeInputs;
pub type AgentResult = Value;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("missing required field: {0}")]
    MissingField(&'static str),
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn execute(&self, inputs: AgentInputs) -> Result<AgentResult, AgentError>;
}

#[derive(Clone)]
pub struct OpenAIAgent {
    name: String,
    model: String,
}

impl OpenAIAgent {
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model: model.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl Agent for OpenAIAgent {
    async fn execute(&self, inputs: AgentInputs) -> Result<AgentResult, AgentError> {
        let messages = inputs
            .get("messages")
            .and_then(Value::as_array)
            .ok_or(AgentError::MissingField("messages"))?;

        let reply = messages
            .iter()
            .filter_map(|message| message.get("content"))
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");

        Ok(json!({
            "model": self.model,
            "reply": reply,
        }))
    }
}

pub fn boxed_agent<A: Agent + 'static>(agent: A) -> Arc<dyn Agent> {
    Arc::new(agent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn openai_agent_concatenates_messages() {
        let agent = OpenAIAgent::new("assistant", "gpt-test");
        let mut inputs = AgentInputs::new();
        inputs.insert(
            "messages".into(),
            Value::Array(vec![
                json!({"role": "user", "content": "Hello"}),
                json!({"role": "system", "content": "world"}),
            ]),
        );

        let result = agent.execute(inputs).await.unwrap();
        assert_eq!(result["reply"], "Hello world");
        assert_eq!(result["model"], "gpt-test");
    }
}
