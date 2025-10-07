mod anthropic;
mod claude_code;
mod crewai;
pub mod framework;
mod gemini_cli;
mod google_adk;
mod google_ai;
mod huggingface;
mod langflow;
mod langsmith;
mod microsoft_ai;
mod openai_codex;
mod salesforce;

use crate::patch::{register_patch, PatchResult};

const DEFAULT_PATCHES: &[(&str, fn() -> PatchResult)] = &[
    ("tygent.integrations.google_ai", google_ai::patch),
    ("tygent.integrations.anthropic", anthropic::patch),
    ("tygent.integrations.huggingface", huggingface::patch),
    ("tygent.integrations.microsoft_ai", microsoft_ai::patch),
    ("tygent.integrations.salesforce", salesforce::patch),
    ("tygent.integrations.claude_code", claude_code::patch),
    ("tygent.integrations.gemini_cli", gemini_cli::patch),
    ("tygent.integrations.openai_codex", openai_codex::patch),
    ("tygent.integrations.crewai", crewai::patch),
    ("tygent.integrations.google_adk", google_adk::patch),
    ("tygent.integrations.langflow", langflow::patch),
    ("tygent.integrations.langsmith", langsmith::patch),
];

pub fn register_default_patches() {
    for (name, patch_fn) in DEFAULT_PATCHES {
        register_patch(*name, *patch_fn);
    }
}

pub fn install_default() -> crate::patch::InstallReport {
    register_default_patches();
    crate::patch::install(None)
}

pub use anthropic::{AnthropicIntegration, AnthropicNode};
pub use framework::langchain::{accelerate_langchain_agent, LangChainAdapter, LangChainAgent};
pub use framework::llamaindex::{
    accelerate_llamaindex_component, LlamaIndexAdapter, LlamaIndexComponent,
};
pub use framework::openai_assistant::{
    accelerate_openai_assistant, OpenAIAssistant, OpenAIAssistantAdapter,
};
pub use google_ai::{GoogleAIIntegration, GoogleAINode};
pub use openai_codex::{OpenAICodexIntegration, OpenAICodexNode};
