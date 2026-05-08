pub mod csv_reader;
pub mod network;
pub mod neuron;
pub mod viz;

#[cfg(test)]
pub mod tests {
    use crate::neuron::{
        Neuron,
        activation::{identity::Identity, step::Step},
        adaline::Adaline,
        classification_config::ClassificationConfig,
        gradient::Gradient,
        perceptron::Perceptron,
        trainable::Trainable,
    };

    #[test]
    fn test_csv_reader() {
        let dataset =
            crate::csv_reader::load_dataset_multi("Datas/Datas/table_3_1.csv", 2, 3, false)
                .expect("Failed to load dataset");

        println!("Dataset loaded: {:?}", dataset);
    }

    #[test]
    fn test_perceptron() {
        let mut p = Perceptron::new(2, 0.0, 1.0, -1.0..=1.0, Step);
        p.zero_weights();
        p.train(
            &[
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 1.0], 0.0),
                (vec![1.0, 0.0], 0.0),
                (vec![1.0, 1.0], 1.0),
            ],
            Some(10000),
        );

        assert_eq!(p.predict(&[0.0, 0.0]), 0.0);
        assert_eq!(p.predict(&[0.0, 1.0]), 0.0);
        assert_eq!(p.predict(&[1.0, 0.0]), 0.0);
        assert_eq!(p.predict(&[1.0, 1.0]), 1.0);
    }

    #[test]
    fn test_gradient_2_1() {
        let mut g = Gradient::new(2, 0.0, 0.2, 0.125001, &None, -1.0..=1.0, Identity);
        g.zero_weights();

        g.train(
            &[
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 1.0], 0.0),
                (vec![1.0, 0.0], 0.0),
                (vec![1.0, 1.0], 1.0),
            ],
            Some(10000),
        );

        assert_eq!(g.classify(&[0.0, 0.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[0.0, 1.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[1.0, 0.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[1.0, 1.0], 0.5, (0.0, 1.0)), 1.0);
    }

    #[test]
    fn test_gradient_2_3() {
        let mut g = Gradient::new(2, 0.0, 0.2, 0.125001, &None, -1.0..=1.0, Identity);
        g.set_weights(vec![1.0, 1.0]);

        g.train(
            &[
                (vec![0.0, 0.0], -1.0),
                (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0),
                (vec![1.0, 1.0], 1.0),
            ],
            Some(10000),
        );
        assert_eq!(g.classify(&[0.0, 0.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[0.0, 1.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[1.0, 0.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[1.0, 1.0], 0.0, (-1.0, 1.0)), 1.0);
    }

    #[test]
    fn test_gradient_2_9() {
        use crate::neuron::gradient::Gradient;

        let mut g = Gradient::new(
            2,
            0.0,
            0.0011,
            0.0,
            &Some(ClassificationConfig {
                error_limit: 0,
                threshold: 0.0,
                values: (-1.0, 1.0),
            }),
            -1.0..=1.0,
            Identity,
        );
        g.zero_weights();

        let dataset = [
            (vec![1.0, 6.0], 1.0),
            (vec![7.0, 9.0], -1.0),
            (vec![1.0, 9.0], 1.0),
            (vec![7.0, 10.0], -1.0),
            (vec![2.0, 5.0], -1.0),
            (vec![2.0, 7.0], 1.0),
            (vec![2.0, 8.0], 1.0),
            (vec![6.0, 8.0], -1.0),
            (vec![6.0, 9.0], -1.0),
            (vec![3.0, 5.0], -1.0),
            (vec![3.0, 6.0], -1.0),
            (vec![3.0, 8.0], 1.0),
            (vec![3.0, 9.0], 1.0),
            (vec![5.0, 7.0], -1.0),
            (vec![5.0, 8.0], -1.0),
            (vec![5.0, 10.0], 1.0),
            (vec![5.0, 11.0], 1.0),
            (vec![4.0, 6.0], -1.0),
            (vec![4.0, 7.0], -1.0),
            (vec![4.0, 9.0], 1.0),
            (vec![4.0, 10.0], 1.0),
        ];

        g.train(&dataset, Some(1000));

        for (input, expected) in dataset {
            assert_eq!(g.classify(&input, 0.0, (-1.0, 1.0)), expected);
        }
    }

    #[test]
    fn test_adeline_2_1() {
        let mut g = Adaline::new(2, 0.0, 0.2, 0.125001, &None, -1.0..=1.0, Identity);
        g.zero_weights();

        g.train(
            &[
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 1.0], 0.0),
                (vec![1.0, 0.0], 0.0),
                (vec![1.0, 1.0], 1.0),
            ],
            Some(10000),
        );

        assert_eq!(g.classify(&[0.0, 0.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[0.0, 1.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[1.0, 0.0], 0.5, (0.0, 1.0)), 0.0);
        assert_eq!(g.classify(&[1.0, 1.0], 0.5, (0.0, 1.0)), 1.0);
    }

    #[test]
    fn test_adeline_2_3() {
        let mut g = Adaline::new(2, 0.0, 0.2, 0.125001, &None, -1.0..=1.0, Identity);
        g.set_weights(vec![1.0, 1.0]);

        g.train(
            &[
                (vec![0.0, 0.0], -1.0),
                (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0),
                (vec![1.0, 1.0], 1.0),
            ],
            Some(10000),
        );
        assert_eq!(g.classify(&[0.0, 0.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[0.0, 1.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[1.0, 0.0], 0.0, (-1.0, 1.0)), -1.0);
        assert_eq!(g.classify(&[1.0, 1.0], 0.0, (-1.0, 1.0)), 1.0);
    }

    #[test]
    fn test_adeline_2_9() {
        let mut g = Adaline::new(
            2,
            0.0,
            0.0011,
            0.0,
            &Some(ClassificationConfig {
                error_limit: 0,
                threshold: 0.0,
                values: (-1.0, 1.0),
            }),
            -1.0..=1.0,
            Identity,
        );
        g.zero_weights();

        let dataset = [
            (vec![1.0, 6.0], 1.0),
            (vec![7.0, 9.0], -1.0),
            (vec![1.0, 9.0], 1.0),
            (vec![7.0, 10.0], -1.0),
            (vec![2.0, 5.0], -1.0),
            (vec![2.0, 7.0], 1.0),
            (vec![2.0, 8.0], 1.0),
            (vec![6.0, 8.0], -1.0),
            (vec![6.0, 9.0], -1.0),
            (vec![3.0, 5.0], -1.0),
            (vec![3.0, 6.0], -1.0),
            (vec![3.0, 8.0], 1.0),
            (vec![3.0, 9.0], 1.0),
            (vec![5.0, 7.0], -1.0),
            (vec![5.0, 8.0], -1.0),
            (vec![5.0, 10.0], 1.0),
            (vec![5.0, 11.0], 1.0),
            (vec![4.0, 6.0], -1.0),
            (vec![4.0, 7.0], -1.0),
            (vec![4.0, 9.0], 1.0),
            (vec![4.0, 10.0], 1.0),
        ];

        g.train(&dataset, Some(1000));

        for (input, expected) in dataset {
            assert_eq!(g.classify(&input, 0.0, (-1.0, 1.0)), expected);
        }
    }
}
