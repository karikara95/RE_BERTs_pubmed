import torch
# x = torch.rand(5, 3)
# print(x)
# print(torch.cuda.is_available())
#from RE_BERTs import main_pretraining, main_task
import main_pretraining, main_task
# bash python -m spacy download en_core_web_lg
# import spacy
# spacy.load('en_core_web_lg')
class args(object):
    def __init__(self,a):
        self.model_no=a["model_no"]
        self.num_classes=a["num_classes"]
        self.model_size = a["model_size"]

arguments={"model_no":0, "num_classes":19, "model_size":'bert-base-uncased'}
A=args(a=arguments)
print(A.model_no)

# print("Getting pairs")
# main_pretraining.main()
# print("Relation Extraction Training")
# main_task.main()

print("Relation Extraction Testing")
from RE_BERTs.src.tasks.infer import infer_from_trained
inferer = infer_from_trained(args=A, detect_entities=False)
test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
#print(inferer.infer_sentence(test, detect_entities=False))

testtweet = "I took [E1]Paxil[/E1] and got [E2]depressed[/E2]."
print(inferer.infer_sentence(testtweet, detect_entities=False))

testpubmed = "This paper presents an unusual complex [E1]suicide[/E1] case that died of [E2]nicotine addiction[/E2]."
print(inferer.infer_sentence(testpubmed, detect_entities=False))


inferer = infer_from_trained(args=A, detect_entities=True)
test2 = "After eating the chicken, he developed a sore throat the next morning."
#print(inferer.infer_sentence(test2, detect_entities=True))