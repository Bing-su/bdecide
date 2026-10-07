use bdecide::hub::{HubOptions, HubOptionsBuilder, ModelSource, Token};
use bdecide::models::clef::{EncodedQuestion, EncodedRecord};
use bdecide::{
    Action,
    Answer,
    ClefConfig,
    Error,
    LayaConfig,
    LoadOptions,
    Metadata,
    ModernBertConfig,
    NoulLabels,
    PredictOptions,
    Question,
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Request,
    Response,
    Truncation,
    Usage,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[test]
fn request_builders_preserve_defaults_and_question_order() {
    // Keep API-created requests equivalent to JSON input, e.g. nonalphabetical IDs.
    let questions: IndexMap<_, _> = ["z", "a"]
        .into_iter()
        .map(|id| {
            (
                id.into(),
                Question::Noul {
                    instructions: "cancel?".into(),
                    criteria: IndexMap::new(),
                    labels: NoulLabels::new(),
                },
            )
        })
        .collect();
    let direct = Request::new(json!("alpha"), questions.clone());
    let built = Request::builder()
        .state(json!("alpha"))
        .questions(questions)
        .build();
    assert_eq!(
        serde_json::to_value(&direct).unwrap(),
        serde_json::to_value(&built).unwrap()
    );
    assert_eq!(
        built
            .questions
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    built.validate().unwrap();
    assert_eq!(
        serde_json::to_value(NoulLabels::builder().build()).unwrap(),
        json!({"false":"false","true":"true"})
    );
    let labels = NoulLabels::builder().r#false("no").r#true("yes").build();
    assert_eq!(labels.r#false, "no");
    assert_eq!(labels.r#true, "yes");
    assert_eq!(
        serde_json::to_value(PredictOptions::builder().build()).unwrap(),
        serde_json::to_value(PredictOptions::new()).unwrap()
    );

    let request = Request::builder()
        .state(json!("alpha"))
        .questions(IndexMap::new())
        .options(
            PredictOptions::builder()
                .truncation(Truncation::Truncate)
                .max_len(3)
                .maybe_head_max_len(Some(16))
                .build(),
        )
        .build();
    assert!(matches!(request.validate(), Err(Error::InvalidRequest(_))));
}

#[test]
fn loading_builders_preserve_constructor_defaults_and_accept_local_sources() {
    // Ensure the new path keeps existing Hub policy, e.g. Auto token selection.
    let direct = HubOptions::new("repo/model");
    // Keep named builder paths public after module moves, e.g. an explicitly typed empty state.
    let builder: HubOptionsBuilder = HubOptions::builder();
    let built = builder.repo_id("repo/model").build();
    assert_eq!(format!("{direct:?}"), format!("{built:?}"));
    let hub = HubOptions::builder()
        .repo_id("repo/model")
        .revision("main")
        .maybe_subfolder(Some("nested"))
        .endpoint("http://localhost")
        .token(Token::Anonymous)
        .local_files_only(true)
        .build();
    assert_eq!(hub.revision.as_deref(), Some("main"));
    assert_eq!(hub.subfolder.as_deref(), Some("nested"));
    assert!(hub.local_files_only);
    assert!(!hub.force_download);
    assert_eq!(
        format!("{:?}", LoadOptions::new("repo/model")),
        format!(
            "{:?}",
            LoadOptions::builder()
                .source(ModelSource::Hub(built))
                .build()
        )
    );
    let local = LoadOptions::builder()
        .source(ModelSource::Local("checkpoint".into()))
        .build();
    assert!(matches!(local.source, ModelSource::Local(_)));
}

#[test]
fn response_and_encoded_record_builders_preserve_payloads() {
    // Exercise public result types together, e.g. a processed noul answer.
    let metadata = Metadata::builder()
        .model_id("local")
        .architecture("clef")
        .device("cpu")
        .build();
    assert_eq!(
        serde_json::to_value(&metadata).unwrap(),
        serde_json::to_value(Metadata::new("local", "clef", "cpu")).unwrap()
    );
    let action = Action::builder().act_probability(0.8).build();
    assert_eq!(
        serde_json::to_value(&action).unwrap(),
        serde_json::to_value(Action::new(0.8)).unwrap()
    );
    let answers = IndexMap::from([(
        "q".into(),
        Answer::Noul {
            noul: 0.9,
            confidence: 0.9,
            answer_confidence: 0.9,
            action,
        },
    )]);
    let direct = Response::new("local", answers.clone(), metadata.clone());
    let built = Response::builder()
        .model("local")
        .answers(answers)
        .metadata(metadata)
        .build();
    assert_eq!(
        serde_json::to_value(direct).unwrap(),
        serde_json::to_value(built).unwrap()
    );
    assert_eq!(
        serde_json::to_value(Usage::builder().build()).unwrap(),
        serde_json::to_value(Usage::new()).unwrap()
    );
    let question = EncodedQuestion::builder()
        .question_id("q")
        .question_type(0)
        .question_span(1..2)
        .option_spans(vec![3..4, 4..5])
        .option_ids(vec!["true".into(), "false".into()])
        .build();
    let direct_question = EncodedQuestion::new(
        "q",
        0,
        1..2,
        vec![3..4, 4..5],
        vec!["true".into(), "false".into()],
    );
    assert_eq!(format!("{question:?}"), format!("{direct_question:?}"));
    let direct = EncodedRecord::new(vec![1, 2, 3, 4, 5], vec![direct_question]);
    let built = EncodedRecord::builder()
        .input_ids(vec![1, 2, 3, 4, 5])
        .questions(vec![question])
        .usage(Usage::builder().input_tokens(5).build())
        .build();
    assert_eq!(built.input_ids, direct.input_ids);
    assert_eq!(
        built.questions[0].option_spans,
        direct.questions[0].option_spans
    );
    assert_eq!(built.usage.input_tokens, 5);
    assert_eq!(direct.usage.input_tokens, 0);
}

#[test]
fn configuration_builders_match_checkpoint_defaults() {
    // Reuse deserialization as an independent default oracle, e.g. missing RoPE settings.
    let text: Qwen3_5TextConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(
        serde_json::to_value(&text).unwrap(),
        serde_json::to_value(Qwen3_5TextConfig::builder().build()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&text).unwrap(),
        serde_json::to_value(Qwen3_5TextConfig::new()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(Qwen3_5Config::builder().build()).unwrap(),
        serde_json::to_value(Qwen3_5Config::new(text)).unwrap()
    );
    let clef = ClefConfig::builder()
        .hidden_size(8)
        .width(8)
        .routing_layers(1)
        .layers(1)
        .heads(2)
        .feedforward(16)
        .build();
    assert_eq!(
        serde_json::to_value(clef).unwrap(),
        serde_json::to_value(ClefConfig::new(8, 8, 1, 1, 2, 16)).unwrap()
    );
    let laya: LayaConfig = serde_json::from_value(json!({"encoder":"modernbert"})).unwrap();
    assert_eq!(
        format!("{laya:?}"),
        format!("{:?}", LayaConfig::builder().encoder("modernbert").build())
    );
    assert_eq!(
        format!("{laya:?}"),
        format!("{:?}", LayaConfig::new("modernbert"))
    );
    let modern: ModernBertConfig = serde_json::from_value(json!({"model_type":"modernbert","hidden_size":8,"intermediate_size":16,"vocab_size":32,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":128})).unwrap();
    let built = ModernBertConfig::builder()
        .hidden_size(8)
        .intermediate_size(16)
        .vocab_size(32)
        .num_hidden_layers(1)
        .num_attention_heads(2)
        .max_position_embeddings(128)
        .build();
    assert_eq!(format!("{modern:?}"), format!("{built:?}"));
    assert_eq!(
        format!("{modern:?}"),
        format!("{:?}", ModernBertConfig::new(8, 16, 32, 1, 2, 128))
    );
    assert!(laya.binning_map.is_null());
    assert_eq!(laya.temperature, vec![Value::from(1.0); 3]);
}

#[cfg(feature = "cpu")]
mod cpu {
    use std::fs;

    use bdecide::{
        AutoModel,
        ClefDecisionModel,
        ClefModel,
        ClefProcessor,
        DecisionModel,
        Error,
        LayaDecisionModel,
        LayaModel,
        Qwen3_5TextModel,
    };
    use burn::backend::Flex;
    use camino::{Utf8Path, Utf8PathBuf};
    use rstest::rstest;

    use super::*;

    fn fixture(name: &str) -> Utf8PathBuf {
        Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[rstest]
    #[case("tiny-laya")]
    #[case("tiny-clef")]
    #[case("tiny-clef-flash")]
    fn model_builders_keep_pretrained_predictions(#[case] name: &str) {
        // Compare with the established loader on bundled weights, e.g. both Clef releases.
        let root = fixture(name);
        let options = || {
            LoadOptions::builder()
                .source(ModelSource::Local(root.clone()))
                .build()
        };
        let direct = AutoModel::from_pretrained(options()).unwrap();
        let via_new = AutoModel::new(options()).unwrap();
        let built = AutoModel::builder().options(options()).build().unwrap();
        let request: Request = serde_json::from_value(
            json!({"state":"alpha", "questions":{"q":{"type":"noul","instructions":"cancel?"}}}),
        )
        .unwrap();
        let expected = serde_json::to_value(direct.predict(&request).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(via_new.predict(&request).unwrap()).unwrap(),
            expected
        );
        assert_eq!(
            serde_json::to_value(built.predict(&request).unwrap()).unwrap(),
            expected
        );
        let device = Default::default();
        // Backend-specific loaders report "flex" where AutoModel reports "cpu";
        // compare each constructor to its own established loader's full response.
        let reference: Box<dyn DecisionModel> = if name == "tiny-laya" {
            Box::new(LayaModel::<Flex>::from_pretrained(&root, &device).unwrap())
        } else {
            Box::new(ClefModel::<Flex>::from_pretrained(&root, &device).unwrap())
        };
        let expected = serde_json::to_value(reference.predict(&request).unwrap()).unwrap();
        let models: Vec<Box<dyn DecisionModel>> = if name == "tiny-laya" {
            vec![
                Box::new(LayaModel::<Flex>::new(&root, &device).unwrap()),
                Box::new(
                    LayaModel::<Flex>::builder()
                        .root(&root)
                        .device(&device)
                        .build()
                        .unwrap(),
                ),
            ]
        } else {
            vec![
                Box::new(ClefModel::<Flex>::new(&root, &device).unwrap()),
                Box::new(
                    ClefModel::<Flex>::builder()
                        .root(&root)
                        .device(&device)
                        .build()
                        .unwrap(),
                ),
            ]
        };
        for model in models {
            assert_eq!(
                serde_json::to_value(model.predict(&request).unwrap()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn processor_builders_keep_encoding_and_model_builders_validate_dimensions() {
        // Route named arguments through initialization checks, e.g. reject zero hidden size.
        let root = fixture("tiny-clef");
        let backbone: Qwen3_5Config =
            serde_json::from_slice(&fs::read(root.join("config.json")).unwrap()).unwrap();
        let direct = ClefProcessor::new(&root, &backbone).unwrap();
        let built = ClefProcessor::builder()
            .root(&root)
            .config(&backbone)
            .build()
            .unwrap();
        let request: Request = serde_json::from_value(
            json!({"state":"alpha", "questions":{"q":{"type":"noul","instructions":"cancel?"}}}),
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", direct.process(&request).unwrap()),
            format!("{:?}", built.process(&request).unwrap())
        );
        let device = Default::default();
        let clef: ClefConfig =
            serde_json::from_slice(&fs::read(root.join("joint_head_config.json")).unwrap())
                .unwrap();
        ClefDecisionModel::<Flex>::new(&clef, &backbone, &device).unwrap();
        ClefDecisionModel::<Flex>::builder()
            .config(&clef)
            .backbone(&backbone)
            .device(&device)
            .build()
            .unwrap();
        Qwen3_5TextModel::<Flex>::new(&backbone.text_config, &device).unwrap();
        Qwen3_5TextModel::<Flex>::builder()
            .config(&backbone.text_config)
            .device(&device)
            .build()
            .unwrap();
        let invalid = Qwen3_5TextConfig::builder().hidden_size(0).build();
        assert!(matches!(
            Qwen3_5TextModel::<Flex>::builder()
                .config(&invalid)
                .device(&device)
                .build(),
            Err(Error::InvalidCheckpoint(_))
        ));
        let invalid_clef = ClefConfig::new(0, 8, 1, 1, 2, 16);
        assert!(matches!(
            ClefDecisionModel::<Flex>::builder()
                .config(&invalid_clef)
                .backbone(&backbone)
                .device(&device)
                .build(),
            Err(Error::InvalidCheckpoint(_))
        ));
        let root = fixture("tiny-laya");
        let laya: LayaConfig =
            serde_json::from_slice(&fs::read(root.join("rl_agent_config.json")).unwrap()).unwrap();
        let encoder: ModernBertConfig =
            serde_json::from_slice(&fs::read(root.join("encoder/config.json")).unwrap()).unwrap();
        LayaDecisionModel::<Flex>::new(&laya, &encoder, &device).unwrap();
        LayaDecisionModel::<Flex>::builder()
            .config(&laya)
            .encoder(&encoder)
            .device(&device)
            .build()
            .unwrap();
        let invalid_encoder = ModernBertConfig::new(0, 16, 32, 1, 2, 128);
        assert!(matches!(
            LayaDecisionModel::<Flex>::builder()
                .config(&laya)
                .encoder(&invalid_encoder)
                .device(&device)
                .build(),
            Err(Error::InvalidCheckpoint(_))
        ));
    }
}
