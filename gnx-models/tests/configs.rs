use gnx_math::{DType, Shape, lax, random::Algorithm};
use gnx_models::{KimiK2Config, KimiK2LM};
use gnx_nn::Rngs;

#[test]
fn full_width_kimi_experts_construct_lazily() {
    // Exercise both block types at checkpoint width without materializing the trillion weights.
    let cfg = KimiK2Config {
        layers: 2,
        ..Default::default()
    };
    let model = KimiK2LM::init(&cfg, DType::BF16, &Rngs::key(0, Algorithm::Philox4x32));
    let experts = model.blocks[1].mlp.experts().unwrap();
    assert_eq!(
        experts.gate.value().shape(),
        Shape::fixed([384, 2048, 7168])
    );
    assert_eq!(
        model.forward(&lax::zeros([1, 1], DType::I32)).shape(),
        Shape::fixed([1, 1, 163840])
    );
}
