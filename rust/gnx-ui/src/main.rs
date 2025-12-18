use dioxus::prelude::*;
use uuid::Uuid;

mod components;
use components::*;

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[layout(NavLayout)]
    #[route("/")]
    Home,
    #[route("/experiment/:id")]
    Experiment { id: Uuid },
}

// const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        // document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: MAIN_CSS },
        Router::<Route> {}
    }
}

#[component]
fn NavLayout() -> Element {
    rsx! {
        div {
            p { "NavLayout" }
            Outlet::<Route> {}
        }
    }
}

/// Home page
#[component]
fn Home() -> Element {
    rsx! {
        "hello, world!"
        Grid {
            Block {
                "Hello, world!"
            }
        }
    }
}

/// Experiment page
#[component]
pub fn Experiment(id: Uuid) -> Element {
    rsx! {}
}


