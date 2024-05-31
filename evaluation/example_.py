from pygaggle.pygaggle.rerank.base import Query, Text
from pygaggle.pygaggle.rerank.transformer import MonoT5

reranker =  MonoT5()

# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# alignscore 0.1.3 requires protobuf<=3.20, but you have protobuf 4.25.3 which is incompatible.
# lens-metric 0.1.1 requires pandas==1.1.5, but you have pandas 2.2.2 which is incompatible.
# lens-metric 0.1.1 requires pytorch-lightning==1.6.0, but you have pytorch-lightning 1.9.5 which is incompatible.
# peft 0.10.0 requires torch>=1.13.0, but you have torch 1.12.1 which is incompatible.
# summac 0.0.4 requires huggingface-hub<=0.17.0, but you have huggingface-hub 0.23.0 which is incompatible.
# torchvision 0.14.1 requires torch==1.13.1, but you have torch 1.12.1 which is incompatible.