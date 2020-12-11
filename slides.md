<img src="../../../figs/logocimat.png" height="20%" width="20%"  align="center"/>

# <center> Métodos de aprendizaje automático para análisis de textos<center>

<center> Víctor Muñiz Sánchez<center>
<center> Diciembre 2020<center>

# Sobre el curso

## Objetivos:

* Mostrar los conceptos básicos de Procesamiento de Lenguaje Natural (NLP) orientado a textos.
* Mostrar repersentaciones vectoriales útiles de textos a partir de modelos probabilísticos de lenguaje y modelos neuronales de lenguaje.
* Abordar modelos de aprendizaje supervisado y no supervisado utilizando métodos de Machine Learning y Deep Learning, dando especial énfasis en arquitecturas de aprendizaje profundo y diversas aplicaciones.
    
    

# Temario

2. Conceptos básicos de python para el curso
3. Introducción y conceptos básicos de NLP
4. Representaciones básicos de textos: one-hot encoding, modelo n-gram, bolsa de palabras y TF-IDF.
5. Embeddings para palabras y documentos basados en modelos neuronales de lenguaje
6. Modelos de ML y aplicaciones en textos
7. Modelos de DL para textos: redes convolucionales, recurrentes y aplicaciones
8. Arquitecturas avanzadas de DL: sequence to sequence y mecanismos de atención (si da tiempo...)

# Introducción

NLP (Jurafsky & Martin, Speech and Language Processing, 2nd. Ed.
Es un campo de estudio enfocado en la interacción entre __lenguaje humano__ y computadoras. Se encuentra en la intersecciónn de ciencias de la computación, inteligencia artificial y linguistica computacional.
    
El objetivo es que las computadoras, realicen tareas útiles que involucren lenguaje humano, como comunicación máquina-humano, mejorar la comunicación humano-humano o simplemente, realizar procesamiento útil de texto o discurso.

Concepto clave: __lenguaje humano__:
\begin{itemize}
\item Signos linguísticos
\item Signos gráficos (textual)
\item Secuencias sonoras
\item Gestos y señales
\end{itemize}

Nosotros hablaremos sobre textos
<img src="../../../figs/noest3.png" height="35%" width="35%" align="center"/>

__NLP es un área bastante compleja__. Esto se debe principalmente, a que el lenguaje natural es complejo en sí:

\begin{itemize}
\item Altamente ambiguo
\item Utiliza procesos mentales complejos para obtener un significado (uso del entorno)
\item Considera diferentes tipos de "entradas": texto, audio, imágenes, expresiones faciales y corporales, otras representaciones pictóricas 😮 👏 🙌 
\item Resultados en tiempo real (machine translation, automatic answering, etc...).
\item En constante evolución
\end{itemize}

También es un área bastante amplia.

¿Cuántas tareas/aplicaciones de NLP para textos conoces?

Veamos lo que dice Wikipedia


```python
import wikipedia
wikipedia.set_lang("en")
print(wikipedia.search("natural language processing"))
```

    ['Natural language processing', 'History of natural language processing', 'Natural language', 'Natural-language understanding', 'Outline of natural language processing', 'Natural-language user interface', 'Natural Language Toolkit', 'Process', 'List of artificial intelligence projects', 'GPT-3']



```python
nlp = wikipedia.page('Natural language processing')
print(nlp.content)
```

    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The result is a computer capable of ‘understanding’ the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. 
    Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural-language generation.
    
    
    == History ==
    
    Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence.
    
    
    === Symbolic NLP (1950s - early 1990s) ===
    The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it is confronted with.
    
    1950s: The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem.  However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced.  Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.
    1960s: Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted "blocks worlds" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966.  Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the "patient" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to "My head hurts" with "Why do you say your head hurts?".
    1970s: During the 1970s, many programmers began to write "conceptual ontologies", which structured real-world information into computer-understandable data.  Examples are MARGIE (Schank, 1975), SAM (Cullingford, 1978), PAM (Wilensky, 1978), TaleSpin (Meehan, 1976), QUALM (Lehnert, 1977), Politics (Carbonell, 1979), and Plot Units (Lehnert 1981).  During this time, the first many chatterbots were written (e.g., PARRY).
    1980s: The 1980s and early 1990s mark the hey-day of symbolic methods in NLP. Focus areas of the time included research on rule-based parsing (e.g., the development of HPSG as a computational operationalization of generative grammar), morphology (e.g., two-level morphology), semantics (e.g., Lesk algorithm), reference (e.g., within Centering Theory) and other areas of natural language understanding (e.g., in the Rhetorical Structure Theory). Other lines of research were continued, e.g., the development of chatterbots with Racter and Jabberwacky. An important development (that eventually led to the statistical turn in the 1990s) was the rising importance of quantitative evaluation in this period.
    
    
    === Statistical NLP (1990s - 2010s) ===
    Up to the 1980s, most natural language processing systems were based on complex sets of hand-written rules.  Starting in the late 1980s, however, there was a revolution in natural language processing with the introduction of machine learning algorithms for language processing.  This was due to both the steady increase in computational power (see Moore's law) and the gradual lessening of the dominance of Chomskyan theories of linguistics (e.g. transformational grammar), whose theoretical underpinnings discouraged the sort of corpus linguistics that underlies the machine-learning approach to language processing.
    1990s: Many of the notable early successes on statistical methods in NLP occurred in the field of machine translation, due especially to work at IBM Research.  These systems were able to take advantage of existing multilingual textual corpora that had been produced by the Parliament of Canada and the European Union as a result of laws calling for the translation of all governmental proceedings into all official languages of the corresponding systems of government.  However, most other systems depended on corpora specifically developed for the tasks implemented by these systems, which was (and often continues to be) a major limitation in the success of these systems. As a result, a great deal of research has gone into methods of more effectively learning from limited amounts of data.
    2000s: With the growth of the web, increasing amounts of raw (unannotated) language data has become available since the mid-1990s. Research has thus increasingly focused on unsupervised and semi-supervised learning algorithms.  Such algorithms can learn from data that has not been hand-annotated with the desired answers or using a combination of annotated and non-annotated data.  Generally, this task is much more difficult than supervised learning, and typically produces less accurate results for a given amount of input data.  However, there is an enormous amount of non-annotated data available (including, among other things, the entire content of the World Wide Web), which can often make up for the inferior results if the algorithm used has a low enough time complexity to be practical.
    
    
    === Neural NLP (present) ===
    In the 2010s, representation learning and deep neural network-style machine learning methods became widespread in natural language processing, due in part to a flurry of results showing that such techniques can achieve state-of-the-art results in many natural language tasks, for example in language modeling, parsing, and many others.
    
    
    == Methods: Rules, statistics, neural networks ==
    In the early days, many language-processing systems were designed by symbolic methods, i.e., the hand-coding of a set of rules, coupled with a dictionary lookup: such as by writing grammars or devising heuristic rules for stemming.
    More recent systems based on machine-learning algorithms have many advantages over hand-produced rules: 
    
    The learning procedures used during machine learning automatically focus on the most common cases, whereas when writing rules by hand it is often not at all obvious where the effort should be directed.
    Automatic learning procedures can make use of statistical inference algorithms to produce models that are robust to unfamiliar input (e.g. containing words or structures that have not been seen before) and to erroneous input (e.g. with misspelled words or words accidentally omitted). Generally, handling such input gracefully with handwritten rules, or, more generally, creating systems of handwritten rules that make soft decisions, is extremely difficult, error-prone and time-consuming.
    Systems based on automatically learning the rules can be made more accurate simply by supplying more input data. However, systems based on handwritten rules can only be made more accurate by increasing the complexity of the rules, which is a much more difficult task. In particular, there is a limit to the complexity of systems based on handwritten rules, beyond which the systems become more and more unmanageable. However, creating more data to input to machine-learning systems simply requires a corresponding increase in the number of man-hours worked, generally without significant increases in the complexity of the annotation process.Despite the popularity of machine learning in NLP research, symbolic methods are still (2020) commonly used
    
    when the amount of training data is insufficient to successfully apply machine learning methods, e.g., for the machine translation of low-resource languages such as provided by the Apertium system,
    for preprocessing in NLP pipelines, e.g., tokenization, or
    for postprocessing and transforming the output of NLP pipelines, e.g., for knowledge extraction from syntactic parses.
    
    
    === Statistical methods ===
    Since the so-called "statistical revolution" in the late 1980s and mid-1990s, much natural language processing research has relied heavily on machine learning. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora (the plural form of corpus, is a set of documents, possibly with human or computer annotations) of typical real-world examples.
    Many different classes of machine-learning algorithms have been applied to natural-language-processing tasks. These algorithms take as input a large set of "features" that are generated from the input data. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
    Some of the earliest-used machine learning algorithms, such as decision trees, produced systems of hard if-then rules similar to existing hand-written rules.  However, part-of-speech tagging introduced the use of hidden Markov models to natural language processing, and increasingly, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to the features making up the input data. The cache language models upon which many speech recognition systems now rely are examples of such statistical models.  Such models are generally more robust when given unfamiliar input, especially input that contains errors (as is very common for real-world data), and produce more reliable results when integrated into a larger system comprising multiple subtasks.
    Since the neural turn, statistical methods in NLP research have been largely replaced by neural networks. However, they continue to be relevant for contexts in which statistical interpretability and transparency is required.
    
    
    === Neural networks ===
    
    A major drawback of statistical methods is that they require elaborate feature engineering. Since the early 2010s, the field has thus largely abandoned statistical methods and shifted to neural networks for machine learning. Popular techniques include the use of word embeddings to capture semantic properties of words, and an increase in end-to-end learning of a higher-level task (e.g., question answering) instead of relying on a pipeline of separate intermediate tasks (e.g., part-of-speech tagging and dependency parsing). In some areas, this shift has entailed substantial changes in how NLP systems are designed, such that deep neural network-based approaches may be viewed as a new paradigm distinct from statistical natural language processing. For instance, the term neural machine translation (NMT) emphasizes the fact that deep learning-based approaches to machine translation directly learn sequence-to-sequence transformations, obviating the need for intermediate steps such as word alignment and language modeling that was used in statistical machine translation (SMT).
    
    
    == Common NLP Tasks ==
    The following is a list of some of the most commonly researched tasks in natural language processing. Some of these tasks have direct real-world applications, while others more commonly serve as subtasks that are used to aid in solving larger tasks.
    Though natural language processing tasks are closely intertwined, they can be subdivided into categories for convenience. A coarse division is given below.
    
    
    === Text and speech processing ===
    Optical character recognition (OCR)
    Given an image representing printed text, determine the corresponding text.Speech recognition
    Given a sound clip of a person or people speaking, determine the textual representation of the speech.  This is the opposite of text to speech and is one of the extremely difficult problems colloquially termed "AI-complete" (see above).  In natural speech there are hardly any pauses between successive words, and thus speech segmentation is a necessary subtask of speech recognition (see below). In most spoken languages, the sounds representing successive letters blend into each other in a process termed coarticulation, so the conversion of the analog signal to discrete characters can be a very difficult process. Also, given that words in the same language are spoken by people with different accents, the speech recognition software must be able to recognize the wide variety of input as being identical to each other in terms of its textual equivalent.
    Speech segmentation
    Given a sound clip of a person or people speaking, separate it into words.  A subtask of speech recognition and typically grouped with it.Text-to-speech
    Given a text, transform those units and produce a spoken representation. Text-to-speech can be used to aid the visually impaired.Word segmentation (Tokenization)
    Separate a chunk of continuous text into separate words. For a language like English, this is fairly trivial, since words are usually separated by spaces. However, some written languages like Chinese, Japanese and Thai do not mark word boundaries in such a fashion, and in those languages text segmentation is a significant task requiring knowledge of the vocabulary and morphology of words in the language. Sometimes this process is also used in cases like bag of words (BOW) creation in data mining.
    
    
    === Morphological analysis ===
    Lemmatization
    The task of removing inflectional endings only and to return the base dictionary form of a word which is also known as a lemma.
    Morphological segmentation
    Separate words into individual morphemes and identify the class of the morphemes. The difficulty of this task depends greatly on the complexity of the morphology (i.e., the structure of words) of the language being considered. English has fairly simple morphology, especially inflectional morphology, and thus it is often possible to ignore this task entirely and simply model all possible forms of a word (e.g., "open, opens, opened, opening") as separate words. In languages such as Turkish or Meitei, a highly agglutinated Indian language, however, such an approach is not possible, as each dictionary entry has thousands of possible word forms.
    Part-of-speech tagging
    Given a sentence, determine the part of speech (POS) for each word. Many words, especially common ones, can serve as multiple parts of speech. For example, "book" can be a noun ("the book on the table") or verb ("to book a flight"); "set" can be a noun, verb or adjective; and "out" can be any of at least five different parts of speech. Some languages have more such ambiguity than others. Languages with little inflectional morphology, such as English, are particularly prone to such ambiguity. Chinese is prone to such ambiguity because it is a tonal language during verbalization. Such inflection is not readily conveyed via the entities employed within the orthography to convey the intended meaning.Stemming
    The process of reducing inflected (or sometimes derived) words to their root form. (e.g., "close" will be the root for "closed", "closing", "close", "closer" etc.).
    
    
    === Syntactic analysis ===
    Grammar induction
    Generate a formal grammar that describes a language's syntax.
    Sentence breaking (also known as "sentence boundary disambiguation")
    Given a chunk of text, find the sentence boundaries. Sentence boundaries are often marked by periods or other punctuation marks, but these same characters can serve other purposes (e.g., marking abbreviations).
    Parsing
    Determine the parse tree (grammatical analysis) of a given sentence. The grammar for natural languages is ambiguous and typical sentences have multiple possible analyses: perhaps surprisingly, for a typical sentence there may be thousands of potential parses (most of which will seem completely nonsensical to a human). There are two primary types of parsing: dependency parsing and constituency parsing. Dependency parsing focuses on the relationships between words in a sentence (marking things like primary objects and predicates), whereas constituency parsing focuses on building out the parse tree using a probabilistic context-free grammar (PCFG) (see also stochastic grammar).
    
    
    === Lexical semantics (of individual words in context) ===
    Lexical semantics
    What is the computational meaning of individual words in context?
    Distributional semantics
    How can we learn semantic representations from data?
    Named entity recognition (NER)
    Given a stream of text, determine which items in the text map to proper names, such as people or places, and what the type of each such name is (e.g. person, location, organization). Although capitalization can aid in recognizing named entities in languages such as English, this information cannot aid in determining the type of named entity, and in any case, is often inaccurate or insufficient.  For example, the first letter of a sentence is also capitalized, and named entities often span several words, only some of which are capitalized.  Furthermore, many other languages in non-Western scripts (e.g. Chinese or Arabic) do not have any capitalization at all, and even languages with capitalization may not consistently use it to distinguish names. For example, German capitalizes all nouns, regardless of whether they are names, and French and Spanish do not capitalize names that serve as adjectives.Sentiment analysis (see also multimodal sentiment analysis)
    Extract subjective information usually from a set of documents, often using online reviews to determine "polarity" about specific objects. It is especially useful for identifying trends of public opinion in social media, for marketing.Terminology extractionThe goal of terminology extraction is to automatically extract relevant terms from a given corpus.
    Word sense disambiguation
    Many words have more than one meaning; we have to select the meaning which makes the most sense in context.  For this problem, we are typically given a list of words and associated word senses, e.g. from a dictionary or an online resource such as WordNet.
    
    
    === Relational semantics (semantics of individual sentences) ===
    Relationship extraction
    Given a chunk of text, identify the relationships among named entities (e.g. who is married to whom).
    Semantic Parsing
    Given a piece of text (typically a sentence), produce a formal representation of its semantics, either as a graph (e.g., in AMR parsing) or in accordance with a logical formalism (e.g., in DRT parsing). This challenge typically includes aspects of several more elementary NLP tasks from semantics (e.g., semantic role labelling, word sense disambiguation) and can be extended to include full-fledged discourse analysis (e.g., discourse analysis, coreference; see Natural Language Understanding below).
    Semantic Role Labelling (see also implicit semantic role labelling below)
    Given a single sentence, identify and disambiguate semantic predicates (e.g., verbal frames), then identify and classify the frame elements (semantic roles).
    
    
    === Discourse (semantics beyond individual sentences) ===
    Coreference resolution
    Given a sentence or larger chunk of text, determine which words ("mentions") refer to the same objects ("entities"). Anaphora resolution is a specific example of this task, and is specifically concerned with matching up pronouns with the nouns or names to which they refer. The more general task of coreference resolution also includes identifying so-called "bridging relationships" involving referring expressions. For example, in a sentence such as "He entered John's house through the front door", "the front door" is a referring expression and the bridging relationship to be identified is the fact that the door being referred to is the front door of John's house (rather than of some other structure that might also be referred to).
    Discourse analysis
    This rubric includes several related tasks.  One task is discourse parsing, i.e., identifying the discourse structure of a connected text, i.e. the nature of the discourse relationships between sentences (e.g. elaboration, explanation, contrast).  Another possible task is recognizing and classifying the speech acts in a chunk of text (e.g. yes-no question, content question, statement, assertion, etc.).Implicit Semantic Role Labelling
    Given a single sentence, identify and disambiguate semantic predicates (e.g., verbal frames) and their explicit semantic roles in the current sentence (see Semantic Role Labelling above). Then, identify semantic roles that are not explicitly realized in the current sentence, classify them into arguments that are explicitly realized elsewhere in the text and those that are not specified, and resolve the former against the local text. A closely related task is zero anaphora resolution, i.e., the extension of coreference resolution to pro-drop languages.Recognizing Textual entailment
    Given two text fragments, determine if one being true entails the other, entails the other's negation, or allows the other to be either true or false.Topic segmentation and recognition
    Given a chunk of text, separate it into segments each of which is devoted to a topic, and identify the topic of the segment.
    
    
    === Higher-level NLP applications ===
    Automatic summarization (text summarization)
    Produce a readable summary of a chunk of text.  Often used to provide summaries of the text of a known type, such as research papers, articles in the financial section of a newspaper.
    Book generation
    Not an NLP task proper but an extension of Natural Language Generation and other NLP tasks is the creation of full-fledged books. The first machine-generated book was created by a rule-based system in 1984 (Racter, The policeman's beard is half-constructed). The first published work by a neural network was published in 2018, 1 the Road, marketed as a novel, contains sixty million words. Both these systems are basically elaborate but non-sensical (semantics-free) language models. The first machine-generated science book was published in 2019 (Beta Writer, Lithium-Ion Batteries, Springer, Cham). Unlike Racter and 1 the Road, this is grounded on factual knowledge and based on text summarization.
    Dialogue management
    Computer systems intended to converse with a human.
    Document AI
    A Document AI platform sits on top of the NLP technology enabling users with no prior experience of artificial intelligence, machine learning or NLP to quickly train a computer to extract the specific data they need from different document types. NLP-powered Document AI enables non-technical teams to quickly access information hidden in documents, for example, lawyers, business analysts and accountants.
    Machine translation
    Automatically translate text from one human language to another.  This is one of the most difficult problems, and is a member of a class of problems colloquially termed "AI-complete", i.e. requiring all of the different types of knowledge that humans possess (grammar, semantics, facts about the real world, etc.) to solve properly.
    Natural language generation (NLG):
    Convert information from computer databases or semantic intents into readable human language.
    Natural language understanding (NLU)
    Convert chunks of text into more formal representations such as first-order logic structures that are easier for computer programs to manipulate. Natural language understanding involves the identification of the intended semantic from the multiple possible semantics which can be derived from a natural language expression which usually takes the form of organized notations of natural language concepts. Introduction and creation of language metamodel and ontology are efficient however empirical solutions. An explicit formalization of natural language semantics without confusions with implicit assumptions such as closed-world assumption (CWA) vs. open-world assumption, or subjective Yes/No vs. objective True/False is expected for the construction of a basis of semantics formalization.
    Question answering
    Given a human-language question, determine its answer.  Typical questions have a specific right answer (such as "What is the capital of Canada?"), but sometimes open-ended questions are also considered (such as "What is the meaning of life?"). Recent works have looked at even more complex questions.
    
    
    == Cognition and NLP ==
    Cognition refers to "the mental action or process of acquiring knowledge and understanding through thought, experience, and the senses." Cognitive science is the interdisciplinary, scientific study of the mind and its processes. Cognitive linguistics is an interdisciplinary branch of linguistics, combining knowledge and research from both psychology and linguistics.  George Lakoff offers a methodology to build Natural language processing (NLP) algorithms through the perspective of Cognitive science, along with the findings of Cognitive linguistics:The first defining aspect of this cognitive task of NLP is the application of the theory of Conceptual metaphor, explained by Lakoff as “the understanding of one idea, in terms of another” which provides an idea of the intent of the author.For example, consider some of the meanings, in English, of the word “big”. When used as a Comparative, as in “That is a big tree,” a likely inference of the intent of the author is that the author is using the word “big” to imply a statement about the tree being ”physically large” in comparison to other trees or the authors experience.  When used as a Stative verb, as in ”Tomorrow is a big day”, a likely inference of the author’s intent it that ”big” is being used to imply ”importance”.  These examples are not presented to be complete, but merely as indicators of the implication of the idea of Conceptual metaphor.  The intent behind other usages, like in ”She is a big person” will remain somewhat ambiguous to a person and a cognitive NLP algorithm alike without additional information.
    This leads to the second defining aspect of this cognitive task of NLP, namely Probabilistic context-free grammar (PCFG) which enables cognitive NLP algorithms to assign relative measures of meaning  to a word, phrase, sentence or piece of text based on the information presented before and after the piece of text being analyzed. The mathematical equation for such algorithms is presented in US patent 9269353 :
    
      
        
          
            
              R
              M
              M
              (
              t
              o
              k
              e
              
                n
                
                  N
                
              
              )
            
            =
            
              P
              M
              M
              (
              t
              o
              k
              e
              
                n
                
                  N
                
              
              )
            
            ×
            
              
                1
                
                  2
                  d
                
              
            
            
              (
              
                
                  ∑
                  
                    i
                    =
                    −
                    d
                  
                  
                    d
                  
                
                
                  (
                  (
                  P
                  M
                  M
                  (
                  t
                  o
                  k
                  e
                  
                    n
                    
                      N
                      −
                      1
                    
                  
                  )
                
                ×
                
                  P
                  F
                  (
                  t
                  o
                  k
                  e
                  
                    n
                    
                      N
                    
                  
                  ,
                  t
                  o
                  k
                  e
                  
                    n
                    
                      N
                      −
                      1
                    
                  
                  )
                  
                    )
                    
                      i
                    
                  
                
              
              )
            
          
        
        {\displaystyle {RMM(token_{N})}={PMM(token_{N})}\times {\frac {1}{2d}}\left(\sum _{i=-d}^{d}{((PMM(token_{N-1})}\times {PF(token_{N},token_{N-1}))_{i}}\right)}
      Where,
         RMM, is the Relative Measure of Meaning
         token, is any block of text, sentence, phrase or word
         N, is the number of tokens being analyzed
         PMM, is the Probable Measure of Meaning based on a corpora
         d, is the location of the token along the sequence of N-1 tokens
         PF, is the Probability Function specific to a language
    
    
    == See also ==
    
    
    == References ==
    
    
    == Further reading ==


Independientemente de la tarea o aplicación, necesitamos una __representación eficiente__ del texto.

*...it is not enough to simply provide a computer with a large amount of data and expect it to learn to speak --the data has to be prepared in such a way that the computer can more easily find patterns and inferences.''*
  
  - James Pustejovsky et al. *Natural Language Annotation.*
  
<img src="../../../figs/anotadores.png" height="35%" width="35%"  align="center"/>

Aquí es donde empezamos con la parte técnica...

# Conceptos básicos de programación para el curso

## Lo que necesitamos saber sobre `python` para éste curso

<img src="../../../figs/python-logo.png" height="35%" width="35%"  align="center"/>

## ¿Porqué `python`?

* Es un lenguaje interpretado (no compilado, como C, C++) y de alto nivel, como Matlab, R o Java.
* Es un lenguaje orientado a objetos
* Es el lenguaje de programación de mayor crecimiento en la última década
* Es el lenguaje científico de mayor uso para aplicaciones de análisis y ciencia de datos, desplazando a R y Matlab
* Es el lenguaje orientado a Deep Learning
* Es "*fácil*" de aprender
* Es `wrapper` de muchas librerias de bajo nivel (C, CUDA-NVIDIA) implementadas en paralelo, lo que mejora el rendimiento

### Python y la infraestructura para cómputo científico y ciencia de datos


<img src="../../../figs/python-libraries.png" height="50%" width="50%" align="center"/>

Veremos algunas cosas básicas y la mayoría, las veremos sobre la marcha.

## Instalación local.

* La opción más óptima: instalar python (https://www.python.org/) y los editores que sean de su preferencia (`vi`, `emacs`, `pycharm`, `spyder`, `jupyter-notebook`, etc).
* La opción más rápida (recomendada para iniciar), instalar Anaconda: https://www.anaconda.com/
* En cualquier caso, recomiendo ampliamente crear __virtual environments__ para proyectos específicos que requieran a su vez, librerías específicas

## Instalar librerías
En Anaconda, se puede hacer directamente en el framework. También puedes hacerlo en consola con los comandos `pip` y `conda` (si tienes Anaconda).

### Ejecución de código python

* The Python interpreter
* The IPython interpreter
* Self-contained Python scripts
* The Jupyter notebook


Jupyter Notebook [*Jupyter Project*](https://jupyter.org/):

* A useful hybrid of the interactive terminal and the self-contained script
* A document format that allows executable code, formatted text, graphics, and even interactive features to be combined into a single document.
* Useful both as a development environment, and as a means of sharing work via rich computational and data-driven narratives that mix together code, figures, data, and text.

### Jupyter + Google Colab

<img src="../../../figs/colab+jupyter.png" height="35%" width="35%" align="center"/>

¿Qué es Colaboratory? (del sitio oficial de Google Colab)

Colaboratory, también llamado Colab, te permite escribir y ejecutar código de Python en un navegador, con las siguientes particularidades:
- Sin configuración requerida
- Acceso gratuito a GPU y TPU
- Facilidad para compartir

Seas <strong>estudiante</strong>, <strong>científico de datos</strong> o <strong>investigador de IA</strong>, Colab facilita tu trabajo. Mira <a href="https://www.youtube.com/watch?v=inN8seMm7UI">este video introductorio sobre Colab</a> para obtener más información.


En este curso, usaremos Jupyter Notebooks y Colab como nuestra plataforma de programación y procesamiento.

¿Qué necesitamos?

* Una cuenta de Google
* Una carpeta dentro de Google Drive para el curso (te ayudará a tenerlo mejor organizado)

El uso de Colab es muy intuitivo, y también hay varios tutoriales para aprender a usarlo. Por ejemplo: 
* [An Absolute Beginner’s Guide To Google Colaboratory](https://medium.com/@dinaelhanan/an-absolute-beginners-guide-to-google-colaboratory-d55c0eb375de).
* [Getting Started With Google Colab](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c)


<img src="../../../figs/github_logo.png" height="45%" width="45%" align="center"/>

El Github para el curso es https://github.com/victorm0202/Curso_banxico/tree/Python-basics

## Python basics

Lo veremos directamente en Colab.

## Volviendo a procesamiento de textos...

Los textos se obtienen de distintas fuentes, tienen diferentes características y su análisis depende en gran parte de tales características.

El procedimiento es como siempre:
\begin{itemize}
\item Obtención del texto
\item Preprocesamiento
\item Representación
\item Modelación
\end{itemize}

En `python` hay varios módulos especializados.

El preproceso y normalización (canonicalización) de texto es una parte muy importante para su posterior modelación y análisis.

Este puede incluir alguno de los siguientes operaciones:

\begin{itemize}
\item Convertir letras a minúsculas o mayúsculas
\item Remover números o convertirlos a palabras
\item Remover signos de puntuación, acentos y otros signos diacríticos
\item Remover caracteres repetidos, incluídos espacios en blanco
\item Remover caracteres con poca frecuencia
\item Remover palabras funcionales o stop words
\item Convertir símbolos a palabras (emojis y otros)
\item Stemming, normalización y POS-tagging
\end{itemize}


Sin embargo, el preproceso y en general, la __extracción__ del texto puede ser mucho más complejo.

Formalidad de la escritura.  The formality continuum (Choudhyrt M. *NLP for social media*)

<img src="../../../figs/formality.png" height="60%" width="60%"  align="center"/>

Datos de la web:

<img src="../../../figs/scrapping.png" height="60%" width="60%"  align="center"/>

Digitalización y reconocimiento de caracteres:

<img src="../../../figs/ocr_error.png" height="40%" width="40%"  align="center"/>

Textos generados por usuarios:

<img src="../../../figs/tuits_ejemplo.png" height="50%" width="50%"  align="center"/>

Textos generados por usuarios:

<img src="../../../figs/sinco.png" height="40%" width="40%"  align="center"/>

Una gran variedad de módulos para preproceso y normalización están disponibles en `python`. Ver por ejemplo:

https://docs.google.com/spreadsheets/d/1-9rMhfcmxFv2V2Q5ZWn1FfLDZZYsuwb1eoSp9CiEEOg/edit#gid=1112515333

Veamos los preprocesamientos básicos que usaremos...


```python
# revisar las funciones, si tienes dudas, me preguntas...
from my_functions import *

preprocesador = preprocesaTexto(idioma='es', _tokeniza=False, _muestraCambios=False, _quitarAcentos=True, 
                                _remueveStop=False, _stemming=False)

txt = 'Hola, me llamo Víctor, vivo en Monterrey, ¿y tú?'
txt_prep = preprocesador.preprocesa(txt)
print(txt_prep)
```

    hola me llamo victor vivo en monterrey y tu



```python
corpus = ['Hola, me llamo Víctor, vivo en Monterrey, ¿y tú?', 'El perro se comió mi tarea.', 
          'Mi vecina se pelea con otra vecina.', 'El gato toca el piano.']
corpus_prep = []
for txt in corpus:
    txt_prep = preprocesador.preprocesa(txt)
    corpus_prep.append(txt_prep)

corpus_prep
```




    ['hola me llamo victor vivo en monterrey y tu',
     'el perro se comio mi tarea',
     'mi vecina se pelea con otra vecina',
     'el gato toca el piano']




```python

```


```python

```


```python

```
