use gnx_math::{DType, lax};
use gnx_models::sphere::{
    BackboneConfig, Decoder, DecoderConfig, Encoder, EncoderConfig, Manifold, PixelHead, spherify,
};
use gnx_nn::{Model, Rngs};

#[test]
fn conditioned_and_register_token_paths_preserve_the_patch_grid() {
    for classes in [0, 3] {
        for manifold in [None, Some(Manifold::Sphere), Some(Manifold::Torus)] {
            let cfg = BackboneConfig {
                manifold,
                classes,
                image_size: 6,
                patch_size: 2,
                token_channels: 5,
                width: 16,
                layers: 3,
                heads: 2,
                context_tokens: 2,
                context_start: 1,
                mixer_depth: 1,
                affine_mixer: false,
                spherify_blocks: true,
                dtype: DType::BF16,
            };
            let enc = Encoder::empty(EncoderConfig {
                backbone: cfg.clone(),
            });
            let dec = Decoder::empty(DecoderConfig {
                pixel_head: PixelHead::Conv,
                backbone: cfg,
            });
            let image = lax::zeros([2, 3, 6, 6], DType::BF16);
            let labels = lax::iota([2], 0, DType::I32);
            let tokens = enc.forward_train(image, &labels, &Rngs::default());
            let shape: &[usize] = if matches!(manifold, Some(Manifold::Torus)) {
                &[2, 9, 5, 2]
            } else {
                &[2, 9, 5]
            };
            assert_eq!(tokens.fixed_dims().unwrap(), shape);
            let out = dec.forward(
                if manifold.is_none() {
                    spherify(&tokens)
                } else {
                    tokens
                },
                None,
            );
            assert_eq!(out.dtype(), DType::BF16);
            assert_eq!(out.fixed_dims().unwrap(), [2, 3, 6, 6]);
        }
    }
}
