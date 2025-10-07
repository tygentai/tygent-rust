use serde_json::Value;

use crate::accelerate::{
    accelerate_framework, AccelerateError, FrameworkExecutor, FrameworkPlanProvider,
};

pub trait LlamaIndexComponent {
    fn llamaindex_plan(&self) -> Option<Value>;
}

#[derive(Clone)]
pub struct LlamaIndexAdapter<T> {
    inner: T,
}

impl<T> LlamaIndexAdapter<T> {
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

impl<T> FrameworkPlanProvider for LlamaIndexAdapter<T>
where
    T: LlamaIndexComponent,
{
    fn framework_plan_value(&self) -> Option<Value> {
        self.inner.llamaindex_plan()
    }
}

pub fn accelerate_llamaindex_component<T>(
    component: T,
) -> Result<FrameworkExecutor<LlamaIndexAdapter<T>>, AccelerateError>
where
    T: LlamaIndexComponent,
{
    accelerate_framework(LlamaIndexAdapter::new(component))
}
