//! Keep fluent System One helpers compatible with ordered JSON requests and answers.
use approx::assert_relative_eq;
use bdecide::Error;
use bdecide::system_one::{
    Metadata,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
};
use rstest::rstest;
use serde_json::{Value, json};

#[test]
fn request_helpers_preserve_json_values_order_and_defaults() {
    let request = SystemOneRequest::builder()
        .state("alpha")
        .build()
        .noul("cancel", "cancel?")
        .choice(
            "route",
            "Route",
            [
                ("billing", json!({"description": "Payments"})),
                ("shipping", json!(true)),
            ],
        )
        .score("severity", "Rate", ["low", "high"])
        .question("extra", SystemOneQuestion::noul(String::from("continue?")));
    request.validate().expect("valid fluent request");
    assert_eq!(
        request
            .questions
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["cancel", "route", "severity", "extra"]
    );
    let value = json!({
        "state":"alpha", "questions":{
            "cancel":{"type":"noul","instructions":"cancel?","criteria":{},"labels":{"false":"false","true":"true"}},
            "route":{"type":"choice","instructions":"Route","criteria":{"billing":{"description":"Payments"},"shipping":true}},
            "severity":{"type":"score","instructions":"Rate","criteria":["low","high"]},
            "extra":{"type":"noul","instructions":"continue?","criteria":{},"labels":{"false":"false","true":"true"}}
        }, "options":{"truncation":"error","max_len":null,"head_max_len":null}
    });
    assert_eq!(serde_json::to_value(&request).expect("JSON"), value);
    let direct = SystemOneRequest::new(String::from("alpha"), request.questions);
    assert_eq!(serde_json::to_value(direct).expect("JSON"), value);
}

#[test]
fn replacing_ids_preserves_positions_and_question_constructors_accept_owned_json() {
    // IDs stay unique as in IndexMap, e.g. replacing the first question keeps it first.
    let request = SystemOneRequest::new(json!({"record": 1}), Default::default())
        .noul("first", "old")
        .noul("second", "continue?")
        .question(
            String::from("first"),
            SystemOneQuestion::choice(
                String::from("Route"),
                [
                    (String::from("b"), json!("old")),
                    (String::from("a"), Value::Null),
                    (String::from("b"), json!({"description":"new"})),
                ],
            ),
        )
        .question(
            "score",
            SystemOneQuestion::score("Rate", [json!(false), json!({"level": 1})]),
        );
    request
        .validate()
        .expect("existing JSON criteria semantics");
    assert_eq!(
        request
            .questions
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["first", "second", "score"]
    );
    let criteria = match request.questions.get("first") {
        Some(SystemOneQuestion::Choice { criteria, .. }) => Some(criteria),
        _ => None,
    }
    .expect("replaced question is a choice");
    assert_eq!(
        criteria.keys().map(String::as_str).collect::<Vec<_>>(),
        ["b", "a"]
    );
    assert_eq!(criteria.get("b"), Some(&json!({"description":"new"})));
    let empty = SystemOneRequest::builder().state(json!([])).build();
    empty.validate().expect("empty questions remain supported");
    let strings = SystemOneQuestion::score("Rate", [String::from("low"), String::from("high")]);
    assert_eq!(
        serde_json::to_value(strings).expect("JSON"),
        json!({"type":"score","instructions":"Rate","criteria":["low","high"]})
    );
}

#[rstest]
#[case::blank_id(SystemOneRequest::new("alpha", Default::default()).noul(" ", "cancel?"))]
#[case::blank_instructions(SystemOneRequest::new("alpha", Default::default()).noul("q", " "))]
#[case::empty_choice(SystemOneRequest::new("alpha", Default::default()).choice("q", "Select", [] as [(&str, Value); 0]))]
#[case::empty_score(SystemOneRequest::new("alpha", Default::default()).score("q", "Rate", [] as [Value; 0]))]
#[case::null_score(SystemOneRequest::new("alpha", Default::default()).score("q", "Rate", [Value::Null]))]
#[case::invalid_state(SystemOneRequest::new(true, Default::default()).noul("q", "cancel?"))]
fn helpers_defer_to_existing_validation(#[case] request: SystemOneRequest) {
    assert!(matches!(request.validate(), Err(Error::InvalidRequest(_))));
}

#[test]
fn response_helpers_read_values_by_id_without_changing_json() {
    let value = json!({
        "model":"local", "answers":{
            "cancel":{"type":"noul","noul":0.8,"confidence":0.6,"answer_confidence":0.8,"action":{"act_probability":0.2}},
            "route":{"type":"choice","choice":"billing","probabilities":{"shipping":0.1,"billing":0.9},"confidence":0.7,"answer_confidence":0.9,"action":{"act_probability":0.3}},
            "severity":{"type":"score","score":1.1,"legend":{"2":"high","0":"low","1":"medium"},"probabilities":{"2":0.2,"0":0.1,"1":0.7},"confidence":0.55,"answer_confidence":0.7,"action":{"act_probability":0.4}}
        },
        "usage":Usage::new(), "metadata":Metadata::new("local", "probe", "cpu")
    });
    let response: SystemOneResponse = serde_json::from_value(value.clone()).expect("response");
    assert_relative_eq!(response.noul("cancel").expect("Noul"), 0.8);
    assert_eq!(response.choice("route"), Some("billing"));
    assert_relative_eq!(response.score("severity").expect("score"), 1.1);
    assert_relative_eq!(
        response
            .choice_probability("route", "billing")
            .expect("choice"),
        0.9
    );
    assert_relative_eq!(
        response
            .score_probability("severity", 2)
            .expect("level by key"),
        0.2
    );

    for (id, confidence, answer_confidence, act_probability) in [
        ("cancel", 0.6, 0.8, 0.2),
        ("route", 0.7, 0.9, 0.3),
        ("severity", 0.55, 0.7, 0.4),
    ] {
        let answer = response.answer(id).expect("existing answer");
        assert_relative_eq!(response.confidence(id).expect("confidence"), confidence);
        assert_relative_eq!(
            response.answer_confidence(id).expect("answer confidence"),
            answer_confidence
        );
        assert_relative_eq!(
            response.act_probability(id).expect("action probability"),
            act_probability
        );
        assert_relative_eq!(
            response.action(id).expect("action").act_probability,
            act_probability
        );
        assert_relative_eq!(answer.confidence(), confidence);
        assert_relative_eq!(answer.answer_confidence(), answer_confidence);
        assert_relative_eq!(answer.act_probability(), act_probability);
        assert!(std::ptr::eq(
            response.action(id).expect("borrowed action"),
            answer.action()
        ));
        if id != "cancel" {
            assert!(answer.noul().is_none());
        }
        if id != "route" {
            assert!(answer.choice().is_none());
            assert!(answer.choice_probability("billing").is_none());
        }
        if id != "severity" {
            assert!(answer.score().is_none());
            assert!(answer.score_probability(2).is_none());
        }
    }
    assert!(response.noul("route").is_none());
    assert!(response.choice("severity").is_none());
    assert!(response.score("cancel").is_none());
    assert!(response.choice_probability("severity", "2").is_none());
    assert!(response.choice_probability("route", "missing").is_none());
    assert!(response.score_probability("route", 0).is_none());
    assert!(response.score_probability("severity", usize::MAX).is_none());
    assert!(response.answer("missing").is_none());
    assert!(response.noul("missing").is_none());
    assert!(response.choice("missing").is_none());
    assert!(response.score("missing").is_none());
    assert!(response.confidence("missing").is_none());
    assert!(response.answer_confidence("missing").is_none());
    assert!(response.action("missing").is_none());
    assert!(response.act_probability("missing").is_none());
    assert!(response.choice_probability("missing", "billing").is_none());
    assert!(response.score_probability("missing", 0).is_none());
    assert_eq!(serde_json::to_value(response).expect("JSON"), value);
}
