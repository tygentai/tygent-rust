//! Framework-specific adapters used by the accelerate decorator.

pub mod langchain;
pub mod llamaindex;
pub mod openai_assistant;

pub use langchain::accelerate_langchain_agent;
pub use llamaindex::accelerate_llamaindex_component;
pub use openai_assistant::accelerate_openai_assistant;
