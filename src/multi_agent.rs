use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::sync::Mutex;

use crate::agent::{boxed_agent, Agent, AgentError, AgentInputs, AgentResult};

#[derive(Clone, Debug)]
pub struct Message {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub content: Value,
    pub timestamp: f64,
}

#[derive(Default)]
pub struct CommunicationBus {
    messages: Mutex<Vec<Message>>,
}

impl CommunicationBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn send(&self, sender: &str, recipient: &str, content: Value) {
        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            from_agent: sender.to_string(),
            to_agent: recipient.to_string(),
            content,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or_default(),
        };

        let mut messages = self.messages.lock().await;
        messages.push(message);
    }

    pub async fn receive(&self, recipient: &str, since: Option<f64>) -> Vec<Message> {
        let messages = self.messages.lock().await;
        messages
            .iter()
            .filter(|msg| msg.to_agent == recipient)
            .filter(|msg| since.map(|t| msg.timestamp > t).unwrap_or(true))
            .cloned()
            .collect()
    }
}

pub struct MultiAgentManager {
    name: String,
    agents: HashMap<String, Arc<dyn Agent>>,
    pub communication_bus: CommunicationBus,
}

impl MultiAgentManager {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            agents: HashMap::new(),
            communication_bus: CommunicationBus::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_agent<A>(&mut self, agent_name: impl Into<String>, agent: A)
    where
        A: Agent + 'static,
    {
        self.agents.insert(agent_name.into(), boxed_agent(agent));
    }

    pub async fn execute(
        &self,
        inputs: AgentInputs,
    ) -> HashMap<String, Result<AgentResult, AgentError>> {
        let mut handles = Vec::new();
        for (name, agent) in &self.agents {
            let agent = agent.clone();
            let agent_inputs = inputs.clone();
            let agent_name = name.clone();
            handles.push(tokio::spawn(async move {
                let result = agent.execute(agent_inputs).await;
                (agent_name, result)
            }));
        }

        let mut outputs = HashMap::new();
        for handle in handles {
            if let Ok((name, result)) = handle.await {
                outputs.insert(name, result);
            }
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;

    #[derive(Clone)]
    struct EchoAgent;

    #[async_trait]
    impl Agent for EchoAgent {
        async fn execute(&self, inputs: AgentInputs) -> Result<AgentResult, AgentError> {
            Ok(json!({ "inputs": inputs }))
        }
    }

    #[tokio::test]
    async fn multi_agent_manager_executes_all_agents() {
        let mut manager = MultiAgentManager::new("team");
        manager.add_agent("one", EchoAgent);
        manager.add_agent("two", EchoAgent);

        let mut inputs = AgentInputs::new();
        inputs.insert("message".into(), Value::String("hello".into()));

        let outputs = manager.execute(inputs).await;
        assert_eq!(outputs.len(), 2);
        assert_eq!(
            outputs.get("one").unwrap().as_ref().unwrap()["inputs"]["message"],
            "hello"
        );
    }
}
