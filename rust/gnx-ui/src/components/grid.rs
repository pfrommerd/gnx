use dioxus::prelude::*;


#[derive(Debug, Clone, PartialEq, Props)]
pub struct GridProps {
    pub children: Element,
}

/// Gridstack-style dashboard component
#[component]
pub fn Grid(props: GridProps) -> Element {
    let height = use_signal(|| 1);
    let width = use_signal(|| 2);
    rsx! {
        div {
            class: "grid",
            style: "--item-height: {height}em; --item-width: {width}em;",
            {props.children}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Props)]
pub struct BlockProps {
    pub children: Element,
}

pub struct DragState {
    // offset from the top left corner of the block
    pub offset_x: f64,
    pub offset_y: f64,
    // the client x, y mouse position
    pub client_x: f64,
    pub client_y: f64,
}

#[component]
pub fn Block(props: BlockProps) -> Element {
    // the pixel size and drag state for the block
    let mut drag_state: Signal<Option<DragState>> = use_signal(|| None);
    let mut size: Signal<(f64, f64)> = use_signal(|| (0., 0.));
    let (width, height) = size.cloned();

    rsx! {
        div {
            class: "grid-item",
            class: if drag_state.read().is_some() { "dragging" },
            style: if let Some(drag) = &*drag_state.read() {
                "left: {drag.client_x - drag.offset_x}px; top: {drag.client_y - drag.offset_y}px; width: {width}px; height: {height}px"
            },
            onresize: move |evt| {
                if let Ok(content) = evt.get_content_box_size() {
                    *size.write() = (content.width, content.height);
                }
            },
            onmousedown: move |evt| {
                // Get the coordinates of the event
                let client_coords = evt.client_coordinates();
                let elem_coords = evt.element_coordinates();
                *drag_state.write() = Some(DragState{
                    offset_x: elem_coords.x,
                    offset_y: elem_coords.y,
                    client_x: client_coords.x,
                    client_y: client_coords.y,
                });
                evt.stop_propagation();
            },
            onmouseup: move |evt| {
                *drag_state.write() = None;
                evt.stop_propagation();
            },
            onmousemove: move |evt| {
                let mut ds = drag_state.write();
                if let Some(ds) = ds.as_mut() {
                    let coords = evt.client_coordinates();
                    ds.client_x = coords.x;
                    ds.client_y = coords.y;
                    evt.stop_propagation();
                }
            },
            {props.children}
        }
        if drag_state.read().is_some() {
            div { class: "grid-item-placeholder", style: "width: {width}px; height: {height}px;" }
        }
    }
}