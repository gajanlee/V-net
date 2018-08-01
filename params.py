class Params:
    # If pickle file has been generated, glove is not neccessary.
    glove_word = "glove.840B.300d.txt"
    vocab_size = 91605
    word_emb_size = 300
    glove_char = "glove.840B.300d.char.txt"
    char_size = 95
    char_emb_size = 300
    emb_pickle = "embeddings.pickle"

    # data directory
    train_path = "dev.json"
    dev_path = "dev.json"
    test_path = "test.json"

    # data content settings
    max_passage_len = 300
    max_question_len = 30
    max_word_len = 8    # every word's max char count

    # model settings
    mode = "train"
    batch_size = {"train": 50, "test": 100}[mode]
    is_training = True if mode == "train" else False
    
    

    
    