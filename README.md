
# LLM Fine-Tuning for US Real Estate Price Prediction with LoRA

This project demonstrates fine-tuning the Qwen-3.0-6B model using LoRA (Low-Rank Adaptation) for real estate data. It includes data exploration, preprocessing, evaluation of base models, and fine-tuning workflows.

## Project Structure

<div align="center">
  <img 
    src="https://www.mermaidchart.com/raw/2f6404e6-2ec5-4064-887d-3fae3a3f39a6?theme=light&version=v0.1&format=svg" 
    alt="Real Estate LLM Fine-Tuning Workflow"
    style="max-width: 100%; height: 2000;"
  />
  <p><em>Figure: End-to-end fine-tuning workflow for Qwen3-0.6B</em></p>
</div>

<img src="docs/workflow-mermaid.svg"/>


```
Qwen-LoRA-Estate/
├── .env                     # Environment secrets
|
├── 1_Data_Exploration_and_Preprocessing.ipynb 
├── 2_Evaluate_Base_Qwen_and_Gemini.ipynb      
├── 3_Qwen3_0_6B_using_LoRA.ipynb              
├── requirements.txt         # Python dependencies
|
├── data/                    # Data directory
│   ├── tabular_data/              # Tabular data
│   │   ├── train_data.csv         
│   │   ├── val_data.csv           
│   ├── text_data/                 # Text data
│   │   ├── dataset-metadata.json  # Metadata for the dataset
│   │   ├── sample_50.jsonl        
│   │   ├── text_train_data.jsonl  
│   │   ├── text_val_data.jsonl    
│   ├── usa-real-estate-dataset/   # Raw dataset
|
├── docs/    # Documentation directory
├── results/ # the evaluation for the base mode, gemini, and fine-tuned model using regression metrics for house price.

```

## Notebooks

1. **1_Data_Exploration_and_Preprocessing.ipynb**  
    This notebook explores the USA Real Estate dataset, checks for inconsistencies, and outlines a plan to prepare it as natural language data for LLM fine-tuning. Text Data generated from this notebook in [Kaggle](https://www.kaggle.com/datasets/hebamo7amed/llm-real-estate-text-data/data) and [Hugging Face](https://huggingface.co/datasets/heba1998/real-estate-data-for-llm-fine-tuning)

2. **2_Evaluate_Base_Qwen_and_Gemini.ipynb**  

   Evaluate the performance of the pretrained base **`Qwen3-0.6B`** model and **`Gemini API`** models on the dataset in order to compare them with the finetuned **`Qwen3-0.6B`** model.

3. **3_Qwen3_0_6B_using_LoRA.ipynb**  
   Fine-tune the **`Qwen3-0.6B`** model using [LoRA](https://arxiv.org/abs/2106.09685) with the **LLaMA-Factory** framework. The goal is to adapt the model for real estate-specific tasks, such as predicting house prices based on structured and natural language data.

## Datasets Description

* **Total Records Available**: \~2,000,000
* **Used for Fine-Tuning**:

  * 5,000 training examples
  * 200 validation examples

> ⚠️ **Note**: Due to infrastructure constraints, the full dataset could not be used in this experiment. Scaling this to larger datasets is part of the future work.

* **Original Dataset**: [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)
* **Converted Dataset**: 

  * Cleans and preprocesses the data (handling missing values, outliers, data types)
  * Translated the full \~2M dataset from tabular Data to natural language hosted in [Kaggle](https://www.kaggle.com/datasets/hebamo7amed/llm-real-estate-text-data/data) and [Hugging Face](https://huggingface.co/datasets/heba1998/real-estate-data-for-llm-fine-tuning)
  * Refactored datasets or training and validation in LlaMa-Factory Style hosted in [Hugging Face](https://huggingface.co/datasets/heba1998/real-estate-data-sample-for-llm-fine-tuning)


## Usage

1. Run the notebooks in sequence:
   - Start with `1_Data_Exploration_and_Preprocessing.ipynb` to explore and prepare the data set .
   - Use `2_Evaluate_Base_Qwen_and_Gemini.ipynb` to evaluate base models.
   - Finally, execute `3_Qwen3_0_6B_using_LoRA.ipynb` to fine-tune the model.

2. Fine-tuned models and results will be saved in the specified output directories.

## Documentation
Additional documentation can be found in the `docs/` directory.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project leverages the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model and [LoRA](https://arxiv.org/abs/2106.09685) for efficient fine-tuning. Special thanks to the contributors and open-source libraries used in this project such as [LlaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) and [vLLM](https://github.com/vllm-project/vllm).
