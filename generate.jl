using Pkg
Pkg.activate(".")
using JSON
using WordTokenizers
using StatsBase
using Flux,CuArrays
using Flux:onehot
using Base.Iterators:partition
using Metalhead
using JLD
using BSON:@save,@load

include("utils.jl")
BASE_PATH = "../../references/mscoco/"
NUM_SENTENCES = 30000
K = 5000

# Load the models
@load "cnn_encoder.bson" cnn_encoder
@load "embedding.bson" embedding
@load "rnn_decoder.bson" rnn_decoder
@load "decoder.bson" decoder

vgg = VGG19() |> gpu
Flux.testmode!(vgg)
vgg = vgg.layers[1:end-3] |> gpu

# Preprocess words #
punc = "!#%&()*+.,-/:;=?@[]^_`{|}~"
punctuation = [punc[i] for i in 1:length(punc)]
data = load_data(BASE_PATH,NUM_SENTENCES,punctuation)
captions = [d[1] for d in data]
tokens = vcat([tokenize(sentence) for sentence in captions]...)
vocab = unique(tokens)
# Sort according to frequencies
freqs = reverse(sort(collect(countmap(tokens)),by=x->x[2]))
top_k_tokens = [freqs[i][1] for i in 1:K]
tokenized_captions = []
for i in 1:length(captions)
    sent_tokens = tokenize(captions[i])
    for j in 1:length(sent_tokens)
        sent_tokens[j] = !(sent_tokens[j] in top_k_tokens) ? "<UNK>" : sent_tokens[j]
    end
    push!(tokenized_captions,sent_tokens)
end
max_length_sentence = maximum([length(cap) for cap in tokenized_captions])
# Pad the sequences
for (i,cap) in enumerate(tokenized_captions)
    if length(cap) < max_length_sentence
        tokenized_captions[i] = [tokenized_captions[i]...,["<PAD>" for i in 1:(max_length_sentence - length(cap))]...]
    end
end
# Define the vocabulary
vocab = [top_k_tokens...,"<UNK>","<PAD>"]
# Define mappings
word2idx = Dict(word=>i for (i,word) in enumerate(vocab))
idx2word = Dict(value=>key for (key,value) in word2idx)
SEQ_LEN = max_length_sentence
# Now - tokenized_captions contains the tokens for each caption in the form of an array

onehotword(word) = Float32.(onehot(word2idx[word],1:length(vocab)))


function reset(rnn_decoder)
    Flux.reset!(rnn_decoder)
end

function sample(image_path)
    img = Metalhead.preprocess(load(image_path)) |> gpu
    features = vgg(img) |> cpu
    
    reset(rnn_decoder)
    prev_word = "<s>"
    lstm_inp = cnn_encoder(features)
    lstm_out = rnn_decoder(lstm_inp)
    output = ""
    
    for i in 1:15
        output = string(output," ",prev_word)
        if prev_word == "</s>"
            break
        end
        word_embeddings = embedding(onehotword(prev_word))
        predictions = softmax(decoder(rnn_decoder(word_embeddings)))
        next_word = idx2word[Flux.argmax(predictions)[1]]
        prev_word = next_word
    end
    
    output
end

image_names = [d[2] for d in data]
println(sample(image_names[1]))
