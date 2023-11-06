#!/bin/bash

fairseq-preprocess     --only-source     --trainpref /lustre/wikitext-103/wiki.train.tokens     --validpref /lustre/wikitext-103/wiki.valid.tokens     --testpref /lustre/wikitext-103/wiki.test.tokens     --destdir /lustre/data/wikitext-103     --workers 96
