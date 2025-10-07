use serde_json::Value;

use crate::accelerate::{
    accelerate_framework, AccelerateError, FrameworkExecutor, FrameworkPlanProvider,
};

pub trait OpenAIAssistant {
    fn openai_plan(&self) -> Option<Value>;
}

#[derive(Clone)]
pub struct OpenAIAssistantAdapter<T> {
    inner: T,
}

impl<T> OpenAIAssistantAdapter<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T> FrameworkPlanProvider for OpenAIAssistantAdapter<T>
where
    T: OpenAIAssistant,
{
    fn framework_plan_value(&self) -> Option<Value> {
        self.inner.openai_plan()
    }
}

pub fn accelerate_openai_assistant<T>(
    assistant: T,
) -> Result<FrameworkExecutor<OpenAIAssistantAdapter<T>>, AccelerateError>
where
    T: OpenAIAssistant,
{
    accelerate_framework(OpenAIAssistantAdapter::new(assistant))
}
