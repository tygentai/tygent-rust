pub mod accelerate;
pub mod adaptive_executor;
pub mod agent;
pub mod audit;
pub mod dag;
pub mod integration;
pub mod multi_agent;
pub mod nodes;
pub mod patch;
pub mod plan_parser;
pub mod prefetch;
pub mod scheduler;
pub mod service_bridge;

pub use accelerate::{
    accelerate, accelerate_framework, accelerate_plan_spec, accelerate_plan_value,
    accelerate_service_plan, AccelerateError, AccelerateKind, Accelerated, FrameworkExecutor,
    FrameworkPlanProvider, PlanExecutor,
};
pub use adaptive_executor::{
    create_conditional_branch_rule, create_fallback_rule, create_resource_adaptation_rule,
    AdaptiveError, AdaptiveExecutionResult, AdaptiveExecutor, ModificationRecord, RewriteRule,
};
pub use agent::{Agent, AgentError, AgentInputs, AgentResult, OpenAIAgent};
pub use audit::{audit_dag, audit_plan, audit_plans, audit_service_plan, AuditError};
pub use dag::{Dag, DagError, EdgeMetadata, EdgeMetadataMap};
pub use integration::{
    install_default as install_default_integrations,
    register_default_patches as register_integration_patches, AnthropicIntegration, AnthropicNode,
    GoogleAIIntegration, GoogleAINode,
};
pub use multi_agent::{CommunicationBus, Message, MultiAgentManager};
pub use nodes::{
    default_llm_latency_model, default_tool_latency_model, LLMNode, Node, NodeExecutionError,
    NodeInputs, NodeMetadata, NodeResult, ToolNode,
};
pub use patch::{install as install_patches, register_patch, InstallReport, PatchResult};
pub use scheduler::{
    ExecutionResults, HookContext, HookOutcome, HookStage, Scheduler, SchedulerError,
};
pub use service_bridge::{
    default_registry, execute_service_plan, LLMRuntimeRegistry, ServiceExecutionError, ServicePlan,
    ServicePlanBuilder, ServicePlanError,
};
