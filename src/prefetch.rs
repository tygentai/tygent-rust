use std::collections::HashMap;
use std::time::Duration;

use tokio::time::sleep;

/// Prefetch resources referenced in the plan.
///
/// This mirrors the Python stub by yielding to the runtime for each URL and
/// returning a map indicating that the link was observed. Real deployments can
/// replace this with an HTTP client and caching layer.
pub async fn prefetch_many<I, S>(links: I) -> HashMap<String, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut results = HashMap::new();

    for link in links {
        let link_str = link.into();
        sleep(Duration::from_millis(0)).await;
        results.insert(link_str, "prefetched".to_string());
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn prefetch_many_marks_links() {
        let links = vec!["https://example.com/a", "https://example.com/b"];
        let results = prefetch_many(links.clone()).await;
        assert_eq!(results.len(), links.len());
        for link in links {
            assert_eq!(results.get(link).unwrap(), "prefetched");
        }
    }

    #[tokio::test]
    async fn prefetch_many_handles_duplicates() {
        let links = vec!["https://example.com/a", "https://example.com/a"];
        let results = prefetch_many(links).await;
        assert_eq!(results.len(), 1);
    }
}
