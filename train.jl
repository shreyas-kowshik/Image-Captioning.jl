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
BASE_PATH = "../data/"

#--------HYPERPARAMETERS----------#
NUM_SENTENCES = 5
# Find top-k tokens
K = 30
BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LOG_FREQUENCY = 10
EPOCHS = 300
SAVE_FREQUENCY = 10
global_step = 1

punc = "!#%&()*+.,-/:;=?@[]^_`{|}~"
punctuation = [punc[i] for i in 1:length(punc)]
data = load_data(BASE_PATH,NUM_SENTENCES,punctuation)

captions = [d[1] for d in data]
tokens = cat([tokenize(sentence) for sentence in captions]...,dims=1)
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
mb_idxs = partition(1:length(data),BATCH_SIZE)
image_names = [d[2] for d in data]

function extract_embedding_features(image_names)
    # extract features from the images and save them to a file
    vgg = VGG19() |> gpu
    Flux.testmode!(vgg)
    vgg = vgg.layers[1:end-3] |> gpu
    
    features = Dict()
    for im_name in image_names
        if im_name in keys(features)
            continue
        end
        
        img = Metalhead.preprocess(load(im_name)) |> gpu
        out = vgg(img)
        
        features[im_name] = out |> cpu
    end
    
    save("features.jld","features",features)
end

function load_embedding_features()
    load("features.jld")["features"]
end

extract_embedding_features(image_names)
features = load_embedding_features()

function get_mb(idx,features)
    cap = tokenized_captions[idx]
    img_names = image_names[idx]
    
    mb_captions = []
    mb_features = []
    mb_targets = []
    
    for i in 1:length(img_names)
         push!(mb_features,features[img_names[i]])
    end
    
    mb_features = hcat(mb_features...)
    # Convert to - Array[SEQ_LEN] with each element - [V,BATCH_SIZE]
    for i in 1:SEQ_LEN
        # Extract and form a batch of each word in sequence
        words = hcat([onehotword(sentence[i]) for sentence in cap]...)
        
        if i < SEQ_LEN
            push!(mb_targets,hcat([onehotword(sentence[i + 1]) for sentence in cap]...))
        else
            push!(mb_targets,hcat([onehotword("<PAD>") for sentence in cap]...))
        end
        
        push!(mb_captions,words)
    end
    
    (mb_captions,mb_features,mb_targets)
end

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

cnn_encoder = Chain(Dense(4096,EMBEDDING_DIM),x->relu.(x))
embedding = Chain(Dense(length(vocab),EMBEDDING_DIM))
rnn_decoder = Chain(LSTM(EMBEDDING_DIM,HIDDEN_DIM))
decoder = Chain(Dense(HIDDEN_DIM,length(vocab)))

function reset()
    Flux.reset!(rnn_decoder.layers[1])
end

function zero_grad_models()
    zero_grad!(cnn_encoder)
    zero_grad!(embedding)
    zero_grad!(rnn_decoder)
    zero_grad!(decoder)
end

function save_models()
    reset()
    @save "cnn_encoder.bson" cnn_encoder
    @save "embedding.bson" embedding
    @save "rnn_decoder.bson" rnn_decoder
    @save "decoder.bson" decoder
end

function load_models()
    @load "cnn_encoder.bson" cnn_encoder
    @load "embedding.bson" embedding
    @load "rnn_decoder.bson" rnn_decoder
    @load "decoder.bson" decoder
end

function get_loss_val(mb_captions,mb_features,mb_targets)
    reset()
    lstm_inp = cnn_encoder(mb_features)
    word_embeddings = embedding.(mb_captions)
    lstm_out = rnn_decoder(lstm_inp)
    predictions = softmax.(decoder.(rnn_decoder.(word_embeddings)))
    sum(Flux.crossentropy.(predictions,mb_targets))
end

model_params = params(params(cnn_encoder)...,params(embedding)...,params(rnn_decoder)...,params(decoder)...)
lr = 1e-4
opt = ADAM(model_params,lr)

for epoch in 1:EPOCHS
    for idx in mb_idxs
        mb_captions,mb_features,mb_targets = get_mb(idx,features)
        zero_grad_models()
        Flux.back!(get_loss_val(mb_captions,mb_features,mb_targets))
        opt()
        global_step += 1
        
        if global_step % LOG_FREQUENCY == 0
            println("---Global Step : $(global_step)")
            println("Loss : $(get_loss_val(mb_captions,mb_features,mb_targets))")
        end
        
        if global_step % SAVE_FREQUENCY == 0
            save_models()
            println("Saved Models!")
        end
    end
end
