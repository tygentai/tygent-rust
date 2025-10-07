use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
use serde_json::Value;

use crate::dag::Dag;
use crate::nodes::NodeInputs;
use crate::scheduler::{Scheduler, SchedulerError};

pub type AdaptiveState = crate::dag::NodeOutputs;

#[derive(Debug, thiserror::Error)]
pub enum AdaptiveError {
    #[error("scheduler execution failed: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("rule '{rule}' action failed: {message}")]
    RuleAction { rule: String, message: String },
}

#[derive(Clone)]
pub struct RewriteRule {
    pub name: String,
    trigger: Arc<dyn Fn(&AdaptiveState) -> bool + Send + Sync>,
    action: Arc<dyn Fn(&Dag, &AdaptiveState) -> Result<Dag, String> + Send + Sync>,
}

impl RewriteRule {
    pub fn new<T, A>(name: impl Into<String>, trigger: T, action: A) -> Self
    where
        T: Fn(&AdaptiveState) -> bool + Send + Sync + 'static,
        A: Fn(&Dag, &AdaptiveState) -> Result<Dag, String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            trigger: Arc::new(trigger),
            action: Arc::new(action),
        }
    }

    pub fn should_apply(&self, state: &AdaptiveState) -> bool {
        (self.trigger)(state)
    }

    pub fn apply(&self, dag: &Dag, state: &AdaptiveState) -> Result<Dag, AdaptiveError> {
        (self.action)(dag, state).map_err(|message| AdaptiveError::RuleAction {
            rule: self.name.clone(),
            message,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ModificationRecord {
    pub rule_name: String,
    pub modification_index: usize,
    pub trigger_state: AdaptiveState,
}

#[derive(Clone)]
pub struct AdaptiveExecutionResult {
    pub results: AdaptiveState,
    pub modification_history: Vec<ModificationRecord>,
    pub final_dag: Dag,
    pub total_modifications: usize,
}

pub struct AdaptiveExecutor {
    base_dag: Dag,
    rewrite_rules: Vec<RewriteRule>,
    max_modifications: usize,
}

impl AdaptiveExecutor {
    pub fn new(base_dag: Dag) -> Self {
        Self {
            base_dag,
            rewrite_rules: Vec::new(),
            max_modifications: 5,
        }
    }

    pub fn with_rules(mut self, rules: Vec<RewriteRule>) -> Self {
        self.rewrite_rules = rules;
        self
    }

    pub fn with_max_modifications(mut self, max: usize) -> Self {
        self.max_modifications = max;
        self
    }

    pub fn add_rule(&mut self, rule: RewriteRule) {
        self.rewrite_rules.push(rule);
    }

    pub async fn execute(
        &self,
        inputs: NodeInputs,
    ) -> Result<AdaptiveExecutionResult, AdaptiveError> {
        let mut current_dag = self.base_dag.clone();
        let mut modification_history = Vec::new();
        let mut total_modifications = 0usize;
        let mut last_results: AdaptiveState = HashMap::new();
        let scheduler_inputs = inputs;

        loop {
            if self.max_modifications > 0 && total_modifications >= self.max_modifications {
                break;
            }

            let mut scheduler = Scheduler::new(current_dag.clone());
            let execution = scheduler.execute(scheduler_inputs.clone()).await?;
            last_results = execution.results.clone();

            let triggered_rule = self
                .rewrite_rules
                .iter()
                .find(|rule| rule.should_apply(&execution.results));

            let Some(rule) = triggered_rule else {
                break;
            };

            let new_dag = rule.apply(&current_dag, &execution.results)?;

            modification_history.push(ModificationRecord {
                rule_name: rule.name.clone(),
                modification_index: total_modifications + 1,
                trigger_state: execution.results.clone(),
            });

            current_dag = new_dag;
            total_modifications += 1;
        }

        Ok(AdaptiveExecutionResult {
            results: last_results,
            modification_history,
            final_dag: current_dag,
            total_modifications,
        })
    }
}

pub fn create_fallback_rule<F, G>(
    error_condition: F,
    fallback_node_creator: G,
    rule_name: impl Into<String>,
) -> RewriteRule
where
    F: Fn(&AdaptiveState) -> bool + Send + Sync + 'static,
    G: Fn(&mut Dag, &AdaptiveState) -> Result<(), String> + Send + Sync + 'static,
{
    RewriteRule::new(rule_name, error_condition, move |dag, state| {
        let mut new_dag = dag.clone();
        fallback_node_creator(&mut new_dag, state)?;
        Ok(new_dag)
    })
}

pub fn create_conditional_branch_rule<F, G>(
    condition: F,
    branch_creator: G,
    rule_name: impl Into<String>,
) -> RewriteRule
where
    F: Fn(&AdaptiveState) -> bool + Send + Sync + 'static,
    G: Fn(&mut Dag, &AdaptiveState) -> Result<(), String> + Send + Sync + 'static,
{
    RewriteRule::new(rule_name, condition, move |dag, state| {
        let mut new_dag = dag.clone();
        branch_creator(&mut new_dag, state)?;
        Ok(new_dag)
    })
}

pub fn create_resource_adaptation_rule<F, G>(
    resource_checker: F,
    adaptation_action: G,
    rule_name: impl Into<String>,
) -> RewriteRule
where
    F: Fn(&AdaptiveState) -> bool + Send + Sync + 'static,
    G: Fn(&Dag, &AdaptiveState) -> Result<Dag, String> + Send + Sync + 'static,
{
    RewriteRule::new(rule_name, resource_checker, adaptation_action)
}

#[cfg(test)]
fn value_as_str<'a>(state: &'a AdaptiveState, node: &str, key: &str) -> Option<&'a str> {
    state
        .get(node)
        .and_then(Value::as_object)
        .and_then(|map| map.get(key))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::ToolNode;
    use serde_json::json;

    fn tool(name: &str, func: impl Fn(&NodeInputs) -> Value + Send + Sync + 'static) -> ToolNode {
        ToolNode::new_sync(name.to_string(), move |inputs| Ok(func(inputs)))
    }

    #[tokio::test]
    async fn fallback_rule_adds_new_node() {
        let mut dag = Dag::new("adaptive");
        dag.add_node(tool("primary", |_| json!({ "status": "error" })));

        let fallback_rule = create_fallback_rule(
            |state| {
                value_as_str(state, "primary", "status") == Some("error")
                    && !state.contains_key("fallback")
            },
            |dag, _state| {
                let fallback = tool("fallback", |_| json!({ "status": "recovered" }));
                dag.add_node(fallback);
                dag.add_edge("primary", "fallback", None)
                    .map_err(|err| err.to_string())?;
                Ok(())
            },
            "fallback",
        );

        let executor = AdaptiveExecutor::new(dag)
            .with_rules(vec![fallback_rule])
            .with_max_modifications(2);
        let result = executor.execute(NodeInputs::new()).await.unwrap();

        assert!(result.results.contains_key("primary"));
        assert!(result.results.contains_key("fallback"));
        assert_eq!(result.total_modifications, 1);
        assert_eq!(result.modification_history.len(), 1);
        assert_eq!(result.modification_history[0].rule_name, "fallback");
    }

    #[tokio::test]
    async fn conditional_branch_rule_skips_when_not_triggered() {
        let mut dag = Dag::new("branch");
        dag.add_node(tool("start", |_| json!({ "value": 1 })));

        let branch_rule = create_conditional_branch_rule(
            |state| value_as_str(state, "start", "value") == Some("2"),
            |dag, _state| {
                let branch = tool("branch", |_| json!({ "branch": true }));
                dag.add_node(branch);
                dag.add_edge("start", "branch", None)
                    .map_err(|err| err.to_string())?;
                Ok(())
            },
            "branch_rule",
        );

        let executor = AdaptiveExecutor::new(dag)
            .with_rules(vec![branch_rule])
            .with_max_modifications(3);
        let result = executor.execute(NodeInputs::new()).await.unwrap();

        assert!(result.results.contains_key("start"));
        assert!(!result.final_dag.has_node("branch"));
        assert_eq!(result.total_modifications, 0);
        assert!(result.modification_history.is_empty());
    }
}
