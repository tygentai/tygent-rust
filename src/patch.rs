use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

pub type PatchResult = Result<(), String>;
type PatchCallback = dyn Fn() -> PatchResult + Send + Sync + 'static;

static PATCH_REGISTRY: Lazy<RwLock<HashMap<String, Arc<PatchCallback>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Default, Debug, Clone)]
pub struct InstallReport {
    pub successes: Vec<String>,
    pub failures: Vec<(String, String)>,
}

pub fn register_patch<F>(name: impl Into<String>, patch: F)
where
    F: Fn() -> PatchResult + Send + Sync + 'static,
{
    let mut registry = PATCH_REGISTRY.write().expect("patch registry poisoned");
    registry.insert(name.into(), Arc::new(patch));
}

pub fn clear_registry() {
    let mut registry = PATCH_REGISTRY.write().expect("patch registry poisoned");
    registry.clear();
}

pub fn install(modules: Option<&[&str]>) -> InstallReport {
    let registry = PATCH_REGISTRY.read().expect("patch registry poisoned");

    let selected: Vec<String> = match modules {
        Some(list) => list.iter().map(|name| name.to_string()).collect(),
        None => registry.keys().cloned().collect(),
    };

    let mut report = InstallReport::default();

    for name in selected {
        match registry.get(&name) {
            Some(patch) => match patch() {
                Ok(()) => report.successes.push(name),
                Err(err) => report.failures.push((name, err)),
            },
            None => report
                .failures
                .push((name.clone(), "patch not registered".into())),
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn register_and_install_patches() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_registry();
        let counter = Arc::new(AtomicUsize::new(0));
        let patch_counter = Arc::clone(&counter);
        register_patch("one", move || {
            patch_counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        register_patch("two", || Err("failure".into()));

        let report = install(None);
        assert_eq!(report.successes, vec!["one".to_string()]);
        assert_eq!(
            report.failures,
            vec![("two".to_string(), "failure".to_string())]
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn install_subset() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_registry();
        register_patch("a", || Ok(()));
        register_patch("b", || Ok(()));
        let report = install(Some(&["a"]));
        assert_eq!(report.successes, vec!["a".to_string()]);
        assert!(report.failures.is_empty());
    }
}
