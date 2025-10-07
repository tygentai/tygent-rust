use serde_json::Value;

use crate::dag::Dag;
use crate::plan_parser::{parse_plan, parse_plans, ParsePlanError, PlanSpec};
use crate::service_bridge::{ServicePlanBuilder, ServicePlanError};

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("plan parsing failed: {0}")]
    Parse(#[from] ParsePlanError),
    #[error("service plan construction failed: {0}")]
    ServicePlan(#[from] ServicePlanError),
}

pub fn audit_dag(dag: &Dag) -> String {
    let mut lines = Vec::new();
    lines.push(format!("DAG: {}", dag.name()));
    for (name, node) in dag.nodes() {
        let dependencies = node
            .dependencies()
            .iter()
            .map(|dep| dep.to_string())
            .collect::<Vec<_>>();
        if dependencies.is_empty() {
            lines.push(format!("- {}: depends on none", name));
        } else {
            lines.push(format!(
                "- {}: depends on {}",
                name,
                dependencies.join(", ")
            ));
        }
    }
    lines.join("\n")
}

pub fn audit_plan(spec: PlanSpec) -> Result<String, AuditError> {
    let (dag, _) = parse_plan(spec)?;
    Ok(audit_dag(&dag))
}

pub fn audit_plans<I>(plans: I) -> Result<String, AuditError>
where
    I: IntoIterator<Item = PlanSpec>,
{
    let (dag, _) = parse_plans(plans)?;
    Ok(audit_dag(&dag))
}

pub fn audit_service_plan(value: &Value) -> Result<String, AuditError> {
    let builder = ServicePlanBuilder::new(None);
    let plan = builder.build(value)?;
    audit_plan(plan.plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn audit_dag_lists_dependencies() {
        let mut dag = Dag::new("demo");
        let node = crate::nodes::ToolNode::new_sync("start", |_| {
            Ok::<_, crate::nodes::NodeExecutionError>(json!(null))
        });
        dag.add_node(node);
        let summary = audit_dag(&dag);
        assert!(summary.contains("DAG: demo"));
        assert!(summary.contains("start"));
    }

    #[test]
    fn audit_service_plan_handles_value() {
        let plan = json!({
            "name": "demo",
            "steps": [
                {
                    "name": "draft",
                    "prompt": "Hello",
                    "metadata": { "provider": "echo" }
                }
            ]
        });
        let summary = audit_service_plan(&plan).unwrap();
        assert!(summary.contains("draft"));
    }
}
