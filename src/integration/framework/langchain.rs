use serde_json::Value;

use crate::accelerate::{
    accelerate_framework, AccelerateError, FrameworkExecutor, FrameworkPlanProvider,
};

pub trait LangChainAgent {
    fn langchain_plan(&self) -> Option<Value>;
}

#[derive(Clone)]
pub struct LangChainAdapter<T> {
    inner: T,
}

impl<T> LangChainAdapter<T> {
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

impl<T> FrameworkPlanProvider for LangChainAdapter<T>
where
    T: LangChainAgent,
{
    fn framework_plan_value(&self) -> Option<Value> {
        self.inner.langchain_plan()
    }
}

pub fn accelerate_langchain_agent<T>(
    agent: T,
) -> Result<FrameworkExecutor<LangChainAdapter<T>>, AccelerateError>
where
    T: LangChainAgent,
{
    accelerate_framework(LangChainAdapter::new(agent))
}
