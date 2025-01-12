class InputFeature(object):
    def __init__(self, input_ids, decoder_input_ids, labels, persona_input_ids, strat_id):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        self.strat_id = strat_id

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        self.persona_input_ids = persona_input_ids
        self.persona_input_length = len(persona_input_ids)
        self.padding_length = max(self.input_length, self.persona_input_length)

        self.input_len = self.input_length + self.decoder_input_length
