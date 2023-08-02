import yaml
from data_gen import DataGen

class Manager():
    def __init__(self, detector_cfg, flashmatch_cfg, photon_library=None):
        self.configure(detector_cfg,  flashmatch_cfg, photon_library)

    def configure(self, detector_cfg, flashmatch_cfg, photon_library):
        config = yaml.load(open(flashmatch_cfg), Loader=yaml.Loader)['FlashMatchManager']
        self.det_cfg = yaml.load(open(detector_cfg), Loader=yaml.Loader)
        self.flash_cfg = yaml.load(open(flashmatch_cfg), Loader=yaml.Loader)

    def make_flashmatch_inputs(self):
        gen = DataGen(self.det_cfg, self.flash_cfg)
        return gen.make_flashmatch_inputs()
    
    def visualize_inputs(self):
        #TODO: Kazu has code to do this
        pass

    def flash_match(self, flashmatch_input):
        #TODO
        pass