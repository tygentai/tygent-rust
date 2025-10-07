use serde_json::Value;

use crate::dag::{Dag, NodeOutputs};
use crate::nodes::NodeInputs;
use crate::plan_parser::{parse_plan, ParsePlanError, PlanSpec};
use crate::prefetch::prefetch_many;
use crate::scheduler::{Scheduler, SchedulerError};
use crate::service_bridge::{ServicePlanBuilder, ServicePlanError};

#[derive(Debug, thiserror::Error)]
pub enum AccelerateError {
    #[error("plan parsing failed: {0}")]
    Parse(ParsePlanError),
    #[error("service plan construction failed: {0}")]
    ServicePlan(ServicePlanError),
    #[error("scheduler execution failed: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("accelerate target unsupported: {0}")]
    Unsupported(&'static str),
}

pub struct PlanExecutor {
    dag: Dag,
    critical: Vec<String>,
    prefetch_links: Vec<String>,
}

impl PlanExecutor {
    pub fn from_spec(spec: PlanSpec, prefetch_links: Vec<String>) -> Result<Self, AccelerateError> {
        let (dag, critical) = parse_plan(spec).map_err(AccelerateError::Parse)?;
        Ok(Self {
            dag,
            critical,
            prefetch_links,
        })
    }

    pub async fn execute(&self, mut inputs: NodeInputs) -> Result<NodeOutputs, AccelerateError> {
        if !self.prefetch_links.is_empty() {
            let prefetched = prefetch_many(self.prefetch_links.clone()).await;
            if !prefetched.is_empty() {
                let mut map = serde_json::Map::new();
                for (key, value) in prefetched {
                    map.insert(key, Value::String(value));
                }
                inputs.insert("prefetch".into(), Value::Object(map));
            }
        }

        let mut scheduler = Scheduler::new(self.dag.clone());
        scheduler.set_priority_nodes(self.critical.clone());
        let results = scheduler.execute(inputs).await?;
        Ok(results.results)
    }
}

pub fn accelerate_service_plan(value: &Value) -> Result<PlanExecutor, AccelerateError> {
    let builder = ServicePlanBuilder::new(None);
    let service_plan = builder.build(value).map_err(AccelerateError::ServicePlan)?;
    PlanExecutor::from_spec(service_plan.plan, service_plan.prefetch_links)
}

pub fn accelerate_plan_spec(spec: &PlanSpec) -> Result<PlanExecutor, AccelerateError> {
    PlanExecutor::from_spec(spec.clone(), Vec::new())
}

pub fn accelerate_plan_value(value: &Value) -> Result<PlanExecutor, AccelerateError> {
    let builder = ServicePlanBuilder::new(None);
    match builder.build(value) {
        Ok(service_plan) => PlanExecutor::from_spec(service_plan.plan, service_plan.prefetch_links),
        Err(err) => Err(AccelerateError::ServicePlan(err)),
    }
}

pub struct FrameworkExecutor<T> {
    original: T,
    plan: PlanExecutor,
}

impl<T> FrameworkExecutor<T> {
    pub fn new(original: T, plan: PlanExecutor) -> Self {
        Self { original, plan }
    }

    pub fn original(&self) -> &T {
        &self.original
    }

    pub fn into_original(self) -> T {
        self.original
    }

    pub async fn execute(&self, inputs: NodeInputs) -> Result<NodeOutputs, AccelerateError> {
        self.plan.execute(inputs).await
    }
}

pub trait FrameworkPlanProvider {
    fn framework_plan_value(&self) -> Option<Value>;
}

pub fn accelerate_framework<T>(framework: T) -> Result<FrameworkExecutor<T>, AccelerateError>
where
    T: FrameworkPlanProvider,
{
    let plan_value = framework
        .framework_plan_value()
        .ok_or(AccelerateError::Unsupported(
            "framework does not expose a plan",
        ))?;
    let plan_executor = accelerate_plan_value(&plan_value)?;
    Ok(FrameworkExecutor::new(framework, plan_executor))
}

pub enum AccelerateKind<'a, T> {
    PlanValue(&'a Value),
    PlanSpec(&'a PlanSpec),
    Framework(T),
}

pub enum Accelerated<T> {
    Plan(PlanExecutor),
    Framework(FrameworkExecutor<T>),
}

pub fn accelerate<T>(target: AccelerateKind<'_, T>) -> Result<Accelerated<T>, AccelerateError>
where
    T: FrameworkPlanProvider,
{
    match target {
        AccelerateKind::PlanValue(value) => accelerate_plan_value(value).map(Accelerated::Plan),
        AccelerateKind::PlanSpec(spec) => accelerate_plan_spec(spec).map(Accelerated::Plan),
        AccelerateKind::Framework(framework) => {
            accelerate_framework(framework).map(Accelerated::Framework)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn executes_accelerated_service_plan() {
        let plan_value = json!({
            "name": "demo",
            "steps": [
                {
                    "name": "draft",
                    "prompt": "Hello {user.name}",
                    "metadata": { "provider": "echo" }
                }
            ]
        });

        let executor = accelerate_service_plan(&plan_value).unwrap();
        let mut inputs = NodeInputs::new();
        let mut user = serde_json::Map::new();
        user.insert("name".into(), Value::String("Ada".into()));
        inputs.insert("user".into(), Value::Object(user));

        let outputs = executor.execute(inputs).await.unwrap();
        let draft = outputs.get("draft").unwrap();
        assert_eq!(
            draft.get("prompt").and_then(Value::as_str).unwrap(),
            "Hello Ada"
        );
    }

    #[test]
    fn accelerate_plan_value_from_json() {
        let plan_value = json!({
            "name": "demo",
            "steps": [
                {
                    "name": "draft",
                    "prompt": "Hello {user}",
                    "metadata": {"provider": "echo"}
                }
            ]
        });
        let plan = accelerate_plan_value(&plan_value).unwrap();
        assert!(plan.prefetch_links.is_empty());
    }

    #[derive(Clone)]
    struct DummyFramework {
        plan: Value,
    }

    impl FrameworkPlanProvider for DummyFramework {
        fn framework_plan_value(&self) -> Option<Value> {
            Some(self.plan.clone())
        }
    }

    #[tokio::test]
    async fn accelerate_framework_yields_executor() {
        let plan = json!({
            "name": "demo",
            "steps": [
                {
                    "name": "step",
                    "prompt": "hi",
                    "metadata": {"provider": "echo"}
                }
            ]
        });
        let framework = DummyFramework { plan };
        let accelerated = accelerate(AccelerateKind::Framework(framework)).unwrap();
        match accelerated {
            Accelerated::Framework(exec) => {
                let inputs = NodeInputs::new();
                let outputs = exec.execute(inputs).await.unwrap();
                assert!(outputs.contains_key("step"));
            }
            Accelerated::Plan(_) => panic!("expected framework executor"),
        }
    }
}
