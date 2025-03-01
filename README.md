# AMIFN

Thank you for visiting this repository. This project proposes Aspect-guided multi-view interactions and fusion network (AMIFN) 
[paper]https://doi.org/10.1016/j.neucom.2023.127222

## Data
- Twitter datasets : the processed pkl files are in floder  `./data/Sentiment_Analysis/twitter201x/` . 
```
data/Sentiment_Analysis/twitter2015: Text data from the twitter2015 dataset
data/Sentiment_Analysis/twitter2017: Text data from the twitter2017 dataset
data/Twitter_image/twitter2015_images: Image data from the twitter2015 dataset
data/Twitter_image/twitter2017_images: Image data from the twitter2017 dataset
```

- Generate    train.pkl     dev.pkl     test.pkl   (look at the file DataProcessor)
- Generate    train.graph   dev.graph   test.graph (look at the file DataProcessor)
- The original tweets, images and sentiment annotations can be downloaded from [https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view]
- Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth),rename to resnet152.pth and put the pre-trained ResNet-152 model under the folder './resnet/resnet" 
- Download roberta-base  https://huggingface.co/roberta-base/tree/main

## Requirement
* torch
* transformers
* tqdm


## Code Usage
 Note that you should use your own data path.
 data_dir 
 imagefeat_dir

```
bash train.sh
```

 ## Acknowledgements
- Most of the codes are based on the codes provided by [huggingface](https://github.com/huggingface/transformers).
- Some code is based on the code of ITM  many thanks!  [ITM](https://github.com/NUSTM/ITM)

