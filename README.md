<h1> Expressions of Anxiety in Political Texts </h1>

Scripts for the study <a href="http://aclweb.org/anthology/W16-5612">"Expressions of Anxiety in Political Texts."</a>  Please cite as:

Rheault, Ludovic. 2016. "Expressions of Anxiety in Political Texts." Proceedings of the 2016 EMNLP Workshop on Natural Language Processing and Computational Social Science.  Austin, Texas: 92-101. 

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

The digitized Canadian Hansard corpus is released publicly in its entirety on the <a href='www.lipad.ca'>www.lipad.ca website</a>. 

<h2> Description of the Files </h2>

<h3> anxiety-classifier.py </h3>

A Python script to test the accuracy of machine learning models using the anxiety score as a feature.  The corpus of sentences annotated for anxiety is the file cf.csv which is expected to be available in the working directory.

<h3> annotation-matcher.py </h3>

A Python script to match the human-coded topic labels used in <a href='http://www.snsoroka.com/data.html'>the Canadian Policy Agendas (Comparative Agenda) project</a> with their original text in the Hansard corpus (www.lipad.ca).  The script requires both datasets to be sorted chronologically.

<h3> anxiety-lexicon.csv </h3>

The anxiety lexicon based on word vectors, including the seed lemmas.

<h3> Coming Soon </h3>

Coming soon.  The dataset for the study of anxiety by topic exceeds the space allocated on GitHub for data files, but will be posted on my personal website and linked here soon.
