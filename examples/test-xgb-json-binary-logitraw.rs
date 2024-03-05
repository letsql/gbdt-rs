extern crate gbdt;

use gbdt::decision_tree::{Data, PredVec, ValueType};
use gbdt::gradient_boost::GBDT;
use gbdt::input;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Use xg.py in xgb-data/xgb_binary_logitraw to generate a model and get prediction results from xgboost.
    // Call this command to convert xgboost model:
    // python examples/convert_xgboost.py xgb-data/xgb_binary_logitraw/xgb.model "binary:logitraw" xgb-data/xgb_binary_logitraw/gbdt.model
    // load model
    let gbdt = GBDT::from_xgboost_json_used_feature("xgb-data/xgb_binary_logitraw/xgb.json")
        .expect("failed to load model");

    // load test data
    let test_file = "xgb-data/xgb_binary_logitraw/agaricus.txt.test";
    let mut input_format = input::InputFormat::txt_format();
    input_format.set_feature_size(126);
    input_format.set_delimeter(' ');
    let test_data = input::load(test_file, input_format).expect("failed to load test data");
    // so given a vector how to transform it
    // if the position is in the keys then add it

    let transformed_test_data = test_data
        .iter()
        .map(|d| {
            Data::new_test_data(
                d.feature
                    .iter()
                    .enumerate()
                    .filter(|(index, &value)| gbdt.feature_mapping.contains_key(&(*index as i64)))
                    .map(|(index, &value)| value)
                    .collect(),
                Some(d.label),
            )
        })
        .collect();

    // inference
    println!("start prediction");
    let predicted: PredVec = gbdt.predict(&transformed_test_data);
    assert_eq!(predicted.len(), test_data.len());

    // compare to xgboost prediction results
    let predict_result = "xgb-data/xgb_binary_logitraw/pred.csv";

    let mut xgb_results = Vec::new();
    let file = File::open(predict_result).expect("failed to load pred.csv");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let text = line.expect("failed to read data from pred.csv");
        let value: ValueType = text.parse().expect("failed to parse data from pred.csv");
        xgb_results.push(value);
    }

    let mut max_diff: ValueType = -1.0;
    for (value1, value2) in predicted.iter().zip(xgb_results.iter()) {
        println!("{} {}", value1, value2);
        let diff = (value1 - value2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!(
        "Compared to results from xgboost, max error is: {:.10}",
        max_diff
    );
    assert!(max_diff < 0.01);
}
