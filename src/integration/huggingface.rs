use std::sync::Arc;

use crate::patch::PatchResult;
use crate::service_bridge::{default_registry, LLMRuntimeError};
use serde_json::Value;

pub fn patch() -> PatchResult {
    let registry = default_registry();
    registry.register(
        "huggingface",
        Arc::new(|_prompt: String, _metadata: Value, _inputs: Value| {
            Box::pin(async {
                Err(LLMRuntimeError::Handler(
                    "huggingface integration not implemented; supply custom executor".into(),
                ))
            })
        }),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_returns_ok() {
        assert!(patch().is_ok());
    }
}
