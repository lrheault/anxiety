<h1> Expressions of Anxiety in Political Texts </h1>

This repo previously contained materials related to the study cited below. 

I've added contents to include a pedagogical example illustrating how to replicate the methods used in the study, which were aimed at creating a lexicon (a "dictionary") to measure *anxiety* in written texts. Given that NLP libraries have improved rapidly since the publication of the paper, I'm using updated resources for this example.

You can follow the steps using the Python notebook named "example.ipynb." This closely replicates the approach used in the study, but the resulting word scores in the anxiety lexicon may be slightly different when recomputing from scratch. The original version of the lexicon is still available in the archive directory. 

To replicate this example, you may first download the corpus here (the zipped file also contains the output of the GloVe program): 

https://drive.google.com/uc?id=1u-lnejm4bzDm7t3YulSowzLaS26IHLpz

After download, continue from the example.ipynb notebook. 

<h2> About </h2>

The study is called <a href="http://aclweb.org/anthology/W16-5612">"Expressions of Anxiety in Political Texts."</a>  Please cite as:

Rheault, Ludovic. 2016. "Expressions of Anxiety in Political Texts." Proceedings of the 2016 EMNLP Workshop on Natural Language Processing and Computational Social Science. Austin, Texas: 92-101. 

<h3> BibTex </h3>
@InProceedings{RHE16, <br>
  author    = {Rheault, Ludovic}, <br>
  title     = {Expressions of Anxiety in Political Texts}, <br>
  booktitle = {Proceedings of the 2016 EMNLP Workshop on Natural Language Processing and Computational Social Science},<br>
  month     = {November},<br>
  year      = {2016},<br>
  address   = {Austin, Texas},<br>
  publisher = {Association for Computational Linguistics},<br>
  pages     = {92--101},<br>
  url       = {http://aclweb.org/anthology/W16-5612}<br>
}<br>

The digitized Canadian Hansard corpus is released publicly in its entirety on the <a href='www.lipad.ca'>www.lipad.ca</a> website. 

<h2> Description of the files in archive/ directory </h2>

<h3> anxiety-classifier.py </h3>

A Python script to test the accuracy of machine learning models using the anxiety score as a feature.  The corpus of sentences annotated for anxiety is the file cf.csv which is expected to be available in the working directory.

<h3> annotation-matcher.py </h3>

A Python script to match the human-coded topic labels used in <a href='http://www.snsoroka.com/data.html'>the Canadian Policy Agendas (Comparative Agenda) project</a> with their original text in the Hansard corpus (www.lipad.ca).  The script requires both datasets to be sorted chronologically.

<h3> anxiety-lexicon.csv </h3>

The anxiety lexicon based on word vectors, including the seed lemmas.

<h3> Dataset </h3>

<a href='http://ludovicrheault.weebly.com/uploads/3/9/4/0/39408253/anxiety-data.tar.gz'>Link to data file and codebook.</a>
