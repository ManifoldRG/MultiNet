class ODinWDefinitions:
    SUB_DATASET_NAMES = {
        'AerialMaritimeDrone':['0', '1', '2', '3', '4'],
        'AmericanSignLanguageLetters': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25'],
        'Aquarium': ['0', '1', '2', '3', '4', '5' ,'6'],
        'BCCD': ['0', '1', '2'],
        'ChessPieces': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12'],
        'CottontailRabbits': ['0'],
        'DroneControl': ['0', '1', '2', '3', '4', '5' ,'6', '7'],
        'EgoHands': ['0', '1', '2', '3'],
        'HardHatWorkers': ['0', '1', '2'],
        'MaskWearing': ['0', '1'],
        'MountainDewCommercial': ['0'],
        'NorthAmericaMushrooms': ['0', '1'],
        'OxfordPets': ['0', '1'],
        'PKLot': ['0', '1'],
        'Packages': ['0'],
        'PascalVOC': [],
        'Raccoon': ['0', '1', '2'],
        'ShellfishOpenImages': ['0', '1', '2'],
        'ThermalCheetah': ['0', '1'],
        'UnoCards': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12', '13', '14'],
        'VehiclesOpenImages': ['0', '1', '2', '3', '4'],
        'WildfireSmoke': ['0'],
        'boggleBoards': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'],
        'brackishUnderwater': ['0', '1', '2', '3', '4', '5'],
        'dice': ['0', '1', '2', '3', '4', '5'],
        'openPoetryVision': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42'],
        'pistols': ['0'],
        'plantdoc': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
        'pothole': ['0'],
        'selfdrivingCar': ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9', '10'],
        'thermalDogsAndPeople': ['0', '1'],
        'vector': ['0'],
        'websiteScreenshots': ['0', '1', '2', '3', '4', '5' ,'6', '7']
        }
    
    SYSTEM_PROMPT = """
    You are a specialized Visual-Language Model Assistant that identifies given object and selects the best option possible
    Read the task carefully. Look at the image and the provided multiple-choice options. 
    Identify the correct object and output only the number corresponding to the correct option.
    Do not output words, category names, or explanations.
    Do not output reasoning steps.
    The answer must be a single integer within the provided range (e.g., 0, 1, 2, …).
    """