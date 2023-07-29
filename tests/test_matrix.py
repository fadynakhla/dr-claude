from dr_claude import datamodels


def test_matrix():
    db = datamodels.DiseaseSymptomKnowledgeBase(
        condition_symptoms={
            datamodels.Condition(name="COVID-19", umls_code="C0000001"): [
                datamodels.WeightedSymptom(
                    name="Fever",
                    umls_code="C0000002",
                    weight=0.5,
                    noise_rate=0.2,
                ),
                datamodels.WeightedSymptom(
                    name="Cough",
                    umls_code="C0000003",
                    weight=0.5,
                    noise_rate=0.1,
                ),
            ],
            datamodels.Condition(name="Common Cold", umls_code="C0000004"): [
                datamodels.WeightedSymptom(
                    name="Fever",
                    umls_code="C0000002",
                    weight=0.5,
                    noise_rate=0.05,
                ),
                datamodels.WeightedSymptom(
                    name="Runny nose",
                    umls_code="C0000004",
                    weight=0.5,
                    noise_rate=0.01,
                ),
            ],
        }
    )

    dataframe, index = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(db)
    assert dataframe.shape == (3, 2)
