//! Verify Decisions wire semantics and local inference, e.g. boolean/string choices never merge.
use std::cell::RefCell;

use approx::assert_relative_eq;
use bdecide::decisions::{
    DecisionChoice,
    DecisionChoiceValue,
    DecisionInputMessage,
    DecisionInputPart,
    DecisionLevel,
    DecisionQuestion,
};
use bdecide::{
    Action,
    DecisionModel,
    DecisionsRequest,
    DecisionsResponse,
    Error,
    Metadata,
    Result,
    SystemOneAnswer,
    SystemOneRequest,
    SystemOneResponse,
};
use indexmap::IndexMap;
use rstest::rstest;
use serde_json::{Value, json};

struct Probe {
    seen: RefCell<Option<SystemOneRequest>>,
    response: SystemOneResponse,
}

impl DecisionModel for Probe {
    fn system_one(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        if request.state == "error" {
            return Err(Error::InvalidRequest("model-specific limit".into()));
        }
        self.seen.replace(Some(request.clone()));
        Ok(self.response.clone())
    }

    fn metadata(&self) -> &Metadata {
        &self.response.metadata
    }
}

fn probe() -> Probe {
    Probe {
        seen: RefCell::new(None),
        response: SystemOneResponse::new(
            "local",
            IndexMap::new(),
            Metadata::new("local", "probe", "cpu"),
        ),
    }
}

fn request(questions: Value) -> DecisionsRequest {
    serde_json::from_value(json!({"model":"local", "input":"alpha", "questions":questions}))
        .expect("valid Decisions request shape")
}

fn predicate() -> SystemOneAnswer {
    SystemOneAnswer::Noul {
        noul: 0.8,
        confidence: 0.6,
        answer_confidence: 0.8,
        action: Action::new(0.2),
    }
}

#[test]
fn request_helpers_preserve_wire_shape_order_and_validation() {
    let request = DecisionsRequest::new("local", "alpha")
        .predicate("cancel", "cancel?")
        .choice("department", "Route", ["billing", "shipping"])
        .choice("approved", "Select", [true, false])
        .choice(
            "typed",
            "Select",
            [
                DecisionChoice::new("true").with_description("literal text"),
                DecisionChoice::new(true),
            ],
        )
        .score(
            "severity",
            "Rate",
            [
                DecisionLevel::new("low").with_description("minor"),
                DecisionLevel::new("high"),
            ],
        )
        .score("simple", "Rate", ["low", "high"])
        .question(DecisionQuestion::predicate("continue?"));
    request.validate().expect("helpers produce a valid request");
    assert_eq!(
        serde_json::to_value(&request).expect("JSON"),
        json!({
            "model":"local","input":"alpha","questions":[
                {"type":"predicate","name":"cancel","instructions":"cancel?"},
                {"type":"choice","name":"department","instructions":"Route","choices":[{"value":"billing"},{"value":"shipping"}]},
                {"type":"choice","name":"approved","instructions":"Select","choices":[{"value":true},{"value":false}]},
                {"type":"choice","name":"typed","instructions":"Select","choices":[{"value":"true","description":"literal text"},{"value":true}]},
                {"type":"score","name":"severity","instructions":"Rate","levels":[{"label":"low","description":"minor"},{"label":"high"}]},
                {"type":"score","name":"simple","instructions":"Rate","levels":[{"label":"low"},{"label":"high"}]},
                {"type":"predicate","instructions":"continue?"}
            ]
        })
    );
    let invalid = DecisionsRequest::new("local", "alpha").choice("q", "Select", [true, true]);
    assert!(matches!(invalid.validate(), Err(Error::InvalidRequest(_))));
    let invalid = DecisionsRequest::new("local", "alpha").predicate("q", " ");
    assert!(matches!(invalid.validate(), Err(Error::InvalidRequest(_))));
    let built = DecisionsRequest::builder()
        .model("local")
        .input("alpha")
        .questions(request.questions.clone())
        .build();
    assert_eq!(
        serde_json::to_value(built).expect("JSON"),
        serde_json::to_value(request).expect("JSON")
    );
}

#[test]
fn message_and_question_helpers_support_owned_and_unnamed_inputs() {
    let request = DecisionsRequest::new(
        "local",
        vec![
            DecisionInputMessage::user("alpha"),
            DecisionInputMessage::user(String::from("beta")),
            DecisionInputMessage::user(vec![
                DecisionInputPart::text("gamma "),
                DecisionInputPart::text("delta"),
            ]),
        ],
    )
    .question(DecisionQuestion::choice(
        "Select",
        [String::from("a"), String::from("b")],
    ))
    .question(
        DecisionQuestion::score("Rate", [String::from("low"), String::from("high")])
            .with_name("old")
            .with_name("new"),
    );
    request.validate().expect("user message helpers");
    assert_eq!(
        serde_json::to_value(request).expect("JSON"),
        json!({
            "model":"local","input":[
                {"role":"user","content":"alpha"},
                {"role":"user","content":"beta"},
                {"role":"user","content":[{"type":"input_text","text":"gamma "},{"type":"input_text","text":"delta"}]}
            ],"questions":[
                {"type":"choice","instructions":"Select","choices":[{"value":"a"},{"value":"b"}]},
                {"type":"score","name":"new","instructions":"Rate","levels":[{"label":"low"},{"label":"high"}]}
            ]
        })
    );
    let owned = DecisionsRequest::new(String::from("local"), String::from("alpha"));
    assert_eq!(
        serde_json::to_value(owned).expect("JSON"),
        json!({"model":"local","input":"alpha","questions":[]})
    );
}

#[test]
fn response_helpers_find_typed_values_without_changing_json() {
    let value = json!({
        "model":"local","answers":[
            {"type":"predicate","name":"cancel","probability":0.8},
            {"type":"choice","name":"department","choice":"billing","confidence":0.7,
             "probabilities":[{"value":"billing","probability":0.9},{"value":"shipping","probability":0.1}]},
            {"type":"choice","name":"approved","choice":true,"confidence":0.6,
             "probabilities":[{"value":"true","probability":0.2},{"value":true,"probability":0.8}]},
            {"type":"score","name":"severity","score":1.1,"confidence":0.55,
             "probabilities":[{"value":2,"label":"high","probability":0.2},{"value":0,"label":"low","probability":0.1},{"value":1,"label":"medium","probability":0.7}]},
            {"type":"refusal","name":"dup"},
            {"type":"predicate","name":"dup","probability":0.6},
            {"type":"predicate","name":null,"probability":0.5},
            {"type":"predicate","name":"","probability":0.4}
        ],"usage":{"input_tokens":42,"output_tokens":0,"total_tokens":42,
                    "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
                    "output_tokens_details":{"reasoning_tokens":0}}
    });
    let response: DecisionsResponse = serde_json::from_value(value.clone()).expect("response");
    assert_relative_eq!(response.probability("cancel").expect("predicate"), 0.8);
    assert_eq!(response.choice_str("department"), Some("billing"));
    assert_eq!(
        response.choice("approved"),
        Some(&DecisionChoiceValue::Boolean(true))
    );
    assert_eq!(response.choice_bool("approved"), Some(true));
    assert!(response.choice_str("approved").is_none());
    assert!(response.choice_bool("department").is_none());
    assert_relative_eq!(response.score("severity").expect("score"), 1.1);
    assert_relative_eq!(response.confidence("department").expect("confidence"), 0.7);
    assert_relative_eq!(response.confidence("severity").expect("confidence"), 0.55);
    assert_relative_eq!(
        response
            .choice_probability("approved", "true")
            .expect("string probability"),
        0.2
    );
    assert_relative_eq!(
        response
            .choice_probability("approved", true)
            .expect("boolean probability"),
        0.8
    );
    assert_relative_eq!(
        response
            .score_probability("severity", 2)
            .expect("level probability"),
        0.2
    );
    assert!(response.choice_probability("approved", false).is_none());
    assert!(response.score_probability("severity", 3).is_none());
    assert!(response.answer("missing").is_none());
    for name in ["missing", "dup"] {
        assert!(response.probability(name).is_none());
        assert!(response.choice(name).is_none());
        assert!(response.score(name).is_none());
        assert!(response.confidence(name).is_none());
        assert!(response.choice_probability(name, true).is_none());
        assert!(response.score_probability(name, 0).is_none());
    }
    assert!(response.probability("department").is_none());
    assert!(response.choice("severity").is_none());
    assert!(response.score("cancel").is_none());
    assert!(response.confidence("cancel").is_none());
    assert!(response.choice_probability("severity", "low").is_none());
    assert!(response.score_probability("department", 0).is_none());
    assert!(
        response
            .answer("dup")
            .expect("first duplicate")
            .is_refusal()
    );
    assert!(!response.answer("cancel").expect("predicate").is_refusal());
    let duplicates: Vec<_> = response
        .answers_named("dup")
        .map(|answer| json!(answer.probability()))
        .collect();
    assert_eq!(duplicates, [Value::Null, json!(0.6)]);
    assert_eq!(response.answers_named("missing").count(), 0);
    let unnamed = response.answers.get(6).expect("unnamed answer");
    assert!(unnamed.name().is_none());
    assert_relative_eq!(unnamed.probability().expect("unnamed predicate"), 0.5);
    assert_relative_eq!(response.probability("").expect("empty name"), 0.4);
    assert_eq!(serde_json::to_value(response).expect("JSON"), value);
}

#[test]
fn typed_choices_names_and_score_follow_wire_contract() {
    let request = request(json!([
        {"type":"predicate", "instructions":"cancel?"},
        {"type":"choice", "name":"same", "instructions":"Select", "choices":[
            {"value":"true", "description":"literal text"}, {"value":true}, {"value":false}
        ]},
        {"type":"score", "name":"same", "instructions":"Rate", "levels":[
            {"label":"low", "description":"minor"}, {"label":"medium"}, {"label":"high"}
        ]},
        {"type":"predicate", "name":"", "instructions":"continue?"}
    ]));
    let mut model = probe();
    // Return answers out of map order to ensure the adapter uses request positions.
    model.response.answers = IndexMap::from([
        (
            "2".into(),
            SystemOneAnswer::Score {
                score: 1.1,
                legend: IndexMap::new(),
                probabilities: IndexMap::from([
                    ("2".into(), 0.2),
                    ("0".into(), 0.1),
                    ("1".into(), 0.7),
                ]),
                confidence: 0.55,
                answer_confidence: 0.7,
                action: Action::new(1.0),
            },
        ),
        (
            "1".into(),
            SystemOneAnswer::Choice {
                choice: "true".into(),
                probabilities: IndexMap::from([
                    ("false".into(), 0.1),
                    ("true".into(), 0.7),
                    ("\"true\"".into(), 0.2),
                ]),
                confidence: 0.6,
                answer_confidence: 0.7,
                action: Action::new(1.0),
            },
        ),
        ("3".into(), predicate()),
        ("0".into(), predicate()),
    ]);
    model.response.usage.input_tokens = 42;
    model.response.usage.output_tokens = 3;
    let value =
        serde_json::to_value(model.decisions(&request).expect("adapter result")).expect("JSON");
    assert_eq!(
        value,
        json!({
            "model":"local", "answers":[
                {"type":"predicate", "name":null, "probability":0.8},
                {"type":"choice", "name":"same", "choice":true, "confidence":0.6,
                 "probabilities":[{"value":"true", "probability":0.2}, {"value":true, "probability":0.7}, {"value":false, "probability":0.1}]},
                {"type":"score", "name":"same", "score":1.1, "confidence":0.55,
                 "probabilities":[{"value":0,"label":"low","probability":0.1}, {"value":1,"label":"medium","probability":0.7}, {"value":2,"label":"high","probability":0.2}]},
                {"type":"predicate", "name":"", "probability":0.8}
            ],
            "usage":{"input_tokens":42,"output_tokens":3,"total_tokens":45,
                     "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
                     "output_tokens_details":{"reasoning_tokens":0}}
        })
    );
    let seen = model.seen.borrow();
    let seen = seen.as_ref().expect("inference called");
    assert_eq!(seen.state, "alpha");
    assert_eq!(
        serde_json::to_value(&seen.questions).expect("JSON"),
        json!({
            "0":{"type":"noul","instructions":"cancel?","criteria":{},"labels":{"false":"false","true":"true"}},
            "1":{"type":"choice","instructions":"Select","criteria":{
                "\"true\"":{"value":"true","description":"literal text"},"true":{"value":true},"false":{"value":false}
            }},
            "2":{"type":"score","instructions":"Rate","criteria":[r#"{"label":"low","description":"minor"}"#,r#"{"label":"medium"}"#,r#"{"label":"high"}"#]},
            "3":{"type":"noul","instructions":"continue?","criteria":{},"labels":{"false":"false","true":"true"}}
        })
    );
    // Hosted refusal responses remain readable, without converting local action thresholds.
    let hosted: DecisionsResponse = serde_json::from_value(json!({
        "model":"hosted","answers":[{"type":"refusal","name":null}],"usage":value.get("usage").expect("usage")
    })).expect("hosted response");
    assert_eq!(
        serde_json::to_value(hosted).expect("JSON").get("answers"),
        Some(&json!([{"type":"refusal","name":null}]))
    );
}

#[test]
fn text_parts_keep_text_order_and_message_boundaries() {
    let mut request = request(json!([]));
    request.input = serde_json::from_value(json!([
        {"role":"user","type":"message","content":[{"type":"input_text","text":"alpha "},{"type":"input_text","text":"beta"}]},
        {"role":"user","content":"gamma"}
    ])).expect("messages");
    let model = probe();
    model.decisions(&request).expect("text messages");
    assert_eq!(
        model.seen.borrow().as_ref().expect("request").state,
        json!([
            {"role":"user","content":"alpha beta"}, {"role":"user","content":"gamma"}
        ])
    );
}

#[rstest]
#[case(json!({"type":"choice","instructions":"Select","choices":[{"value":"a"}]}))]
#[case(json!({"type":"choice","instructions":"Select","choices":[{"value":true},{"value":true}]}))]
#[case(json!({"type":"score","instructions":"Rate","levels":[]}))]
#[case(json!({"type":"predicate","instructions":" "}))]
fn rejects_invalid_questions_before_inference(#[case] question: Value) {
    let request = request(json!([question]));
    let model = probe();
    assert!(matches!(request.validate(), Err(Error::InvalidRequest(_))));
    assert!(matches!(
        model.decisions(&request),
        Err(Error::InvalidRequest(_))
    ));
    assert!(model.seen.borrow().is_none());
}

#[test]
fn rejects_model_mismatch_images_and_oversized_metadata_before_inference() {
    let mut request = request(json!([]));
    let model = probe();
    request.model = "another-model".into();
    assert!(matches!(
        model.decisions(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.model = "local".into();
    request.safety_identifier = Some("한".repeat(129));
    assert!(matches!(
        model.decisions(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.safety_identifier = Some("한".repeat(128));
    request.validate().expect("128 Unicode characters");
    request.input = serde_json::from_value(json!([{"role":"user","content":[
        {"type":"input_image","image_url":"https://example.com/image.png","detail":null}
    ]}]))
    .expect("image shape");
    assert!(
        matches!(model.decisions(&request), Err(Error::InvalidRequest(message)) if message.contains("text only"))
    );
    assert!(model.seen.borrow().is_none());
}

#[rstest]
#[case(json!([{"role":"system","content":"alpha"}]))]
#[case(json!([{"role":"user","type":"function_call","content":"alpha"}]))]
#[case(json!([{"role":"user","content":[{"type":"input_audio","audio":"x"}]}]))]
fn rejects_unsupported_input_shapes(#[case] input: Value) {
    serde_json::from_value::<DecisionsRequest>(
        json!({"model":"local","input":input,"questions":[]}),
    )
    .expect_err("unsupported input must fail to deserialize");
}

#[test]
fn numeric_choice_values_are_not_in_the_wire_union() {
    serde_json::from_value::<DecisionsRequest>(json!({
        "model":"local","input":"alpha","questions":[{"type":"choice","instructions":"Select","choices":[{"value":1},{"value":2}]}]
    })).expect_err("numeric choice values are unsupported");
}

#[test]
fn choice_count_bounds_are_two_through_255() {
    for count in [0, 1, 2, 255, 256] {
        let choices: Vec<_> = (0..count)
            .map(|index| json!({"value":index.to_string()}))
            .collect();
        let request = request(json!([{"type":"choice","instructions":"Select","choices":choices}]));
        if (2..=255).contains(&count) {
            request.validate().expect("allowed option count");
        } else {
            assert!(matches!(request.validate(), Err(Error::InvalidRequest(_))));
        }
    }
}

#[test]
fn malformed_model_answers_fail_instead_of_dropping_results() {
    let request = request(json!([{"type":"predicate","instructions":"cancel?"}]));
    let mut model = probe();
    assert!(matches!(
        model.decisions(&request),
        Err(Error::Inference(_))
    ));
    model
        .response
        .answers
        .insert("wrong-id".into(), predicate());
    assert!(matches!(
        model.decisions(&request),
        Err(Error::Inference(_))
    ));
    model.response.answers = IndexMap::from([("0".into(), predicate())]);
    model.response.usage.input_tokens = usize::MAX;
    model.response.usage.output_tokens = 1;
    assert!(matches!(
        model.decisions(&request),
        Err(Error::Inference(_))
    ));
}

#[cfg(any(feature = "cpu", feature = "wgpu"))]
fn infer(fixture: &str, device: bdecide::Device) {
    use bdecide::hub::ModelSource;
    use bdecide::{AutoModel, LoadOptions};

    let model = AutoModel::from_pretrained(
        LoadOptions::builder()
            .source(ModelSource::Local(
                format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR")).into(),
            ))
            .device(device)
            .build(),
    )
    .expect("fixture model");
    let mut request = request(json!([
        {"type":"predicate","instructions":"cancel?"},
        {"type":"choice","name":"q","instructions":"Select","choices":[{"value":"alpha"},{"value":true}]},
        {"type":"score","name":"q","instructions":"Rate","levels":[{"label":"low"},{"label":"high"}]}
    ]));
    request.model = model.metadata().model_id.clone();
    let response = model.decisions(&request).expect("Decisions inference");
    assert_eq!(response.answers.len(), 3);
    assert!(response.usage.input_tokens > 0);
    assert_eq!(response.usage.total_tokens, response.usage.input_tokens);
    let value = serde_json::to_value(&response).expect("JSON");
    let answers = value
        .get("answers")
        .and_then(Value::as_array)
        .expect("answers");
    for answer in answers {
        if let Some(probabilities) = answer.get("probabilities").and_then(Value::as_array) {
            let sum: f64 = probabilities
                .iter()
                .map(|entry| {
                    entry
                        .get("probability")
                        .and_then(Value::as_f64)
                        .expect("probability")
                })
                .sum();
            assert!((sum - 1.0).abs() < 1e-6);
            if let Some(score) = answer.get("score").and_then(Value::as_f64) {
                let expected: f64 = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| {
                        index as f64
                            * entry
                                .get("probability")
                                .and_then(Value::as_f64)
                                .expect("probability")
                    })
                    .sum();
                assert!((score - expected).abs() < 1e-6);
            }
        }
    }
    assert_eq!(
        answers
            .get(1)
            .expect("choice")
            .get("probabilities")
            .expect("probabilities")
            .as_array()
            .expect("array")
            .get(1)
            .expect("boolean option")
            .get("value"),
        Some(&json!(true))
    );
    // Compare the complete schema; Clef jointly attends to all question heads.
    let direct: SystemOneRequest = serde_json::from_value(json!({"state":"alpha","questions":{
        "0":{"type":"noul","instructions":"cancel?"},
        "1":{"type":"choice","instructions":"Select","criteria":{
            "\"alpha\"":{"value":"alpha"},"true":{"value":true}
        }},
        "2":{"type":"score","instructions":"Rate","criteria":[r#"{"label":"low"}"#,r#"{"label":"high"}"#]}
    }}))
    .expect("System One request");
    let direct = model.system_one(&direct).expect("System One inference");
    let direct = serde_json::to_value(&direct.answers).expect("System One JSON");
    let noul = direct
        .get("0")
        .and_then(|answer| answer.get("noul"))
        .and_then(Value::as_f64)
        .expect("Noul probability");
    let probability = answers
        .first()
        .expect("predicate")
        .get("probability")
        .and_then(Value::as_f64)
        .expect("probability");
    assert!((probability - noul).abs() < 1e-6);
    // Verify Score too, e.g. the supplied rubric must match direct text-criteria inference.
    for (index, keys) in [(1, ["\"alpha\"", "true"]), (2, ["0", "1"])] {
        let probabilities = direct
            .get(index.to_string())
            .and_then(|answer| answer.get("probabilities"))
            .expect("direct probabilities");
        let actual = answers
            .get(index)
            .expect("answer")
            .get("probabilities")
            .and_then(Value::as_array)
            .expect("probabilities");
        for (entry, key) in actual.iter().zip(keys) {
            let actual = entry
                .get("probability")
                .and_then(Value::as_f64)
                .expect("probability");
            assert!(
                (actual
                    - probabilities
                        .get(key)
                        .and_then(Value::as_f64)
                        .expect("probability"))
                .abs()
                    < 1e-6
            );
        }
    }
}

#[test]
fn model_errors_propagate_without_becoming_refusals() {
    let mut request = request(json!([{"type":"predicate","instructions":"cancel?"}]));
    request.input = bdecide::decisions::DecisionInput::Text("error".into());
    assert!(
        matches!(probe().decisions(&request), Err(Error::InvalidRequest(message)) if message == "model-specific limit")
    );
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-laya")]
#[case("tiny-clef")]
#[case("tiny-vev-4b")]
#[case("tiny-wald")]
#[case("tiny-decider")]
#[case("tiny-von")]
#[case("tiny-d1-3b")]
#[case("tiny-d1-omni")]
fn cpu_decisions_runs_local_models(#[case] fixture: &str) {
    infer(fixture, bdecide::Device::flex());
}

#[cfg(feature = "wgpu")]
#[rstest]
#[case("tiny-laya")]
#[case("tiny-von")]
#[case("tiny-wald")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_decisions_runs_local_model(#[case] fixture: &str) {
    infer(fixture, bdecide::Device::wgpu(Default::default()));
}
