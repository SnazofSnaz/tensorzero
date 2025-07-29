use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("NVIDIA_API_KEY") {
        Ok(key) => HashMap::from([("nvidia_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim-extra-body".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim-extra-headers".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim-dynamic".to_string(),
        model_name: "nvidia_nim-dynamic".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: credentials.clone(),
    }];

    let tool_use_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: credentials.clone(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "nvidia-nim".to_string(),
            model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
            model_provider_name: "nvidia_nim".to_string(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "nvidia-nim-strict".to_string(),
            model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
            model_provider_name: "nvidia_nim".to_string(),
            credentials: credentials.clone(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia_nim_json_mode_off".to_string(),
        model_name: "nvidia_nim/meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: credentials.clone(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim-shorthand".to_string(),
        model_name: "nvidia_nim::meta/llama-3.1-8b-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "nvidia-nim-vision".to_string(),
        model_name: "nvidia_nim/microsoft/phi-3.5-vision-instruct".to_string(),
        model_provider_name: "nvidia_nim".to_string(),
        credentials: credentials.clone(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![], // NVIDIA NIM doesn't have reasoning models like o1
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: tool_use_providers.clone(),
        tool_multi_turn_inference: tool_use_providers.clone(),
        dynamic_tool_use_inference: tool_use_providers.clone(),
        parallel_tool_use_inference: tool_use_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers,
        pdf_inference: vec![], // NVIDIA NIM may not support PDF inference directly
        shorthand_inference: shorthand_providers.clone(),
    }
}
