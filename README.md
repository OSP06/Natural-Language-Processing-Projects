NLP Spelling Corrector (Edit Distance Based)
This repository contains Python code demonstrating a basic spelling correction and word suggestion system using edit distance algorithms. This project was developed as part of an NLP (Natural Language Processing) class to explore fundamental concepts in text processing and string manipulation.

Project Overview
The core idea behind this project is to suggest corrections for misspelled words by finding words in a given vocabulary that are a certain "edit distance" away from the misspelled input. An "edit" refers to a single character operation (deletion, insertion, replacement, or transposition/switch).

Features
Four Basic Edit Operations:

delete_char: Generates words by deleting one character.

switch_char: Generates words by swapping adjacent characters.

replace_char: Generates words by replacing one character with another.

insert_char: Generates words by inserting a character.

Edit Distance Calculation:

edit_distance_one: Finds all valid words (from a vocabulary) that are one edit away from an input word.

edit_distance_two: Finds all valid words (from a vocabulary) that are two edits away from an input word.

Spelling Suggestion Engine (fix_edits):

Checks if a word is already correct.

Prioritizes suggestions that are one edit away.

Falls back to suggestions that are two edits away if no one-edit suggestions are found.

Assigns a basic probability to suggestions (currently based on inverse vocabulary size, which could be extended with actual word frequencies).

Misspelled Word Finder (find_misspelled_words):

Reads a text file and a vocabulary file.

Identifies words in the text file that are not present in the vocabulary.

Provides spelling suggestions for each identified misspelled word.

Getting Started
Prerequisites
Python 3.x

No external libraries beyond collections (Counter, defaultdict), re (for regex), and string (for ascii_lowercase), which are standard.

Installation
No specific installation steps are required. Simply download the Team_Group1_NLP.ipynb notebook.

Usage
Open the Jupyter Notebook: It is recommended to run this notebook in a Jupyter environment (e.g., Google Colab, Anaconda JupyterLab).

Upload Sample Files: You will need two text files for the find_misspelled_words example:

shakespeare.txt: The text file to be checked for misspellings.

THE SONNETS.txt: A vocabulary list (used as the 'correct' word dictionary).
(Note: These files are referenced as /content/shakespeare.txt and /content/THE SONNETS.txt in the notebook's example usage, assuming a Google Colab environment. Adjust paths if running locally.)

Run Cells: Execute the cells sequentially. The notebook demonstrates:

Individual string manipulation functions.

edit_distance_one and edit_distance_two examples.

The fix_edits function for single word suggestions.

The find_misspelled_words function to process a larger text and generate suggestions for all detected misspellings.

Code Structure
The notebook is divided into logical sections:

String Manipulation: Functions for delete_char, switch_char, replace_char, insert_char.

Edit Distance Calculations: edit_distance_one and edit_distance_two.

Edit Distances with Spelling Suggestions: Contains the Read function for vocabulary, re-definitions of the basic edit functions for clarity, fix_edits for spelling suggestions, and find_misspelled_words for a full text analysis.

Limitations & Possible Enhancements
Simple Probability Model: The current probability model (e.g., 1 / len(vocab)) is very basic. A more advanced system would use actual word frequencies from a large corpus to assign more realistic probabilities to suggestions.

Vocabulary: The effectiveness heavily depends on the quality and comprehensiveness of the vocab_file.

Performance: For very large vocabularies or long words, calculating edit_distance_two can be computationally intensive.

Contextual Spelling: This system is purely based on edit distance and does not consider the context of the word in a sentence (e.g., distinguishing "there" from "their").

Punctuation Handling: The re.findall regex for words can be refined to handle various punctuation scenarios more robustly.

Potential Enhancements:

Implement a more sophisticated probability model using collections.Counter on a larger corpus to get actual word frequencies.

Add a user interface (e.g., using ipywidgets in Jupyter) to interactively test words.

Explore more advanced spelling correction techniques (e.g., Norvig's approach, SymSpell, or transformer-based models for contextual correction).
