class ODinWDefinitions:
    SUB_DATASET_CATEGORIES = {
        'AerialMaritimeDrone': 5,
        'AmericanSignLanguageLetters': 26,
        'Aquarium': 7,
        'BCCD': 3,
        'ChessPieces': 13,
        # 'CottontailRabbits': 1,
        'DroneControl': 8,
        'EgoHands': 4,
        'HardHatWorkers': 3,
        'MaskWearing': 2,
        # 'MountainDewCommercial': 1,
        'NorthAmericaMushrooms': 2,
        'OxfordPets': 37,
        'PKLot': 2,
        # 'Packages': 1,
        # 'Raccoon': 1,
        'ShellfishOpenImages': 3,
        'ThermalCheetah': 2,
        'UnoCards': 15,
        'VehiclesOpenImages': 5,
        # 'WildfireSmoke': 1,
        'boggleBoards': 36,
        'brackishUnderwater': 6,
        'dice': 6,
        'openPoetryVision': 43,
        # 'pistols': 1,
        'plantdoc': 30,
        # 'pothole': 1,
        'selfdrivingCar': 11,
        'thermalDogsAndPeople': 2,
        # 'vector': 1,
        'websiteScreenshots': 8,
        }
    
    SYSTEM_PROMPT = """
        You are a specialized Visual-Language Model Assistant that identifies the object in a given image and selects the best option possible from the options provided.
        Read the task carefully. Look at the image and the provided multiple-choice options.
        Identify the correct object and output ONLY the number corresponding to the correct option.
        The answer must be a single integer within the provided range of options (e.g., 0, 1, 2, …).
        Do not output words, category names, or explanations.
        Do not output reasoning steps.
    """