extern crate gbdt;

use gbdt::decision_tree::ValueType;
use gbdt::gradient_boost::GBDT;
use gbdt::input;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Use xg.py in xgb-data/xgb_multi_softmax to generate a model and get prediction results from xgboost.
    // Call this command to convert xgboost model:
    // python examples/convert_xgboost.py xgb-data/xgb_multi_softmax/xgb.model "multi:softmax" xgb-data/xgb_multi_softmax/gbdt.model
    // load model
    let gbdt = GBDT::from_xgboost_json("xgb-data/xgb_multi_softmax/xgb.json")
        .expect("failed to load model");

    // load test data
    let test_file = "xgb-data/xgb_multi_softmax/dermatology.data.test";
    let mut input_format = input::InputFormat::csv_format();
    input_format.set_label_index(34);
    let test_data = input::load(test_file, input_format).expect("failed to load test data");

    // inference
    println!("start prediction");
    let (labels, _probs) = gbdt.predict_multiclass(&test_data, 6);
    assert_eq!(labels.len(), test_data.len());

    // compare to xgboost prediction results
    let predict_result = "xgb-data/xgb_multi_softmax/pred.csv";

    let mut xgb_results = Vec::new();
    let file = File::open(predict_result).expect("failed to load pred.csv");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let text = line.expect("failed to read data from pred.csv");
        let value: ValueType = text.parse().expect("failed to parse data from pred.csv");
        xgb_results.push(value);
    }

    let mut max_diff: ValueType = -1.0;
    for (value1, value2) in labels.iter().zip(xgb_results.iter()) {
        println!("{} {}", value1, value2);
        let diff = (*value1 as ValueType - *value2).abs();
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
