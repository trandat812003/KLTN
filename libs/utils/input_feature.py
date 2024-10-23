class InputFeatures:
    def __init__(self, input_ids, decoder_input_ids, labels):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        
        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length