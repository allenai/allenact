"""Common constants used when training agents to complete tasks in iTHOR, the
interactive version of AI2-THOR.
"""

from collections import OrderedDict
from typing import Set, Tuple, Dict

MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_DOWN = "LookDown"
LOOK_UP = "LookUp"
END = "End"

VISIBILITY_DISTANCE = 1.25
FOV = 90.0

ORDERED_SCENE_TYPES = ("kitchens", "livingrooms", "bedrooms", "bathrooms")

NUM_SCENE_TYPES = len(ORDERED_SCENE_TYPES)


def make_scene_name(type_ind, scene_num):
    if type_ind == 1:
        return "FloorPlan" + str(scene_num) + "_physics"
    elif scene_num < 10:
        return "FloorPlan" + str(type_ind) + "0" + str(scene_num) + "_physics"
    else:
        return "FloorPlan" + str(type_ind) + str(scene_num) + "_physics"


SCENES_TYPE_TO_SCENE_NAMES = OrderedDict(
    [
        (
            ORDERED_SCENE_TYPES[type_ind - 1],
            tuple(
                make_scene_name(type_ind=type_ind, scene_num=scene_num)
                for scene_num in range(1, 31)
            ),
        )
        for type_ind in range(1, NUM_SCENE_TYPES + 1)
    ]
)

SCENES_TYPE_TO_TRAIN_SCENE_NAMES = OrderedDict(
    (key, scenes[:20]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

SCENES_TYPE_TO_VALID_SCENE_NAMES = OrderedDict(
    (key, scenes[20:25]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

SCENES_TYPE_TO_TEST_SCENE_NAMES = OrderedDict(
    (key, scenes[25:30]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

ALL_SCENE_NAMES = sum(SCENES_TYPE_TO_SCENE_NAMES.values(), tuple())

TRAIN_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_TRAIN_SCENE_NAMES.values()), tuple()
)

VALID_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_VALID_SCENE_NAMES.values()), tuple()
)
TEST_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_TEST_SCENE_NAMES.values()), tuple()
)

TRAIN_SCENE_NAMES_SET = set(TRAIN_SCENE_NAMES)
VALID_SCENE_NAMES_SET = set(VALID_SCENE_NAMES)
TEST_SCENE_NAMES_SET = set(TEST_SCENE_NAMES)

_object_type_and_location_tsv = """
AlarmClock	bedrooms
Apple	kitchens
ArmChair	livingrooms,bedrooms
BaseballBat	bedrooms
BasketBall	bedrooms
Bathtub	bathrooms
BathtubBasin	bathrooms
Bed	bedrooms
Blinds	kitchens,bedrooms
Book	kitchens,livingrooms,bedrooms
Boots	livingrooms,bedrooms
Bottle	kitchens
Bowl	kitchens,livingrooms,bedrooms
Box	livingrooms,bedrooms
Bread	kitchens
ButterKnife	kitchens
Cabinet	kitchens,livingrooms,bedrooms,bathrooms
Candle	livingrooms,bathrooms
Cart	bathrooms
CD	bedrooms
CellPhone	kitchens,livingrooms,bedrooms
Chair	kitchens,livingrooms,bedrooms
Cloth	bedrooms,bathrooms
CoffeeMachine	kitchens
CoffeeTable	livingrooms,bedrooms
CounterTop	kitchens,livingrooms,bedrooms,bathrooms
CreditCard	kitchens,livingrooms,bedrooms
Cup	kitchens
Curtains	kitchens,livingrooms,bedrooms
Desk	bedrooms
DeskLamp	livingrooms,bedrooms
DiningTable	kitchens,livingrooms,bedrooms
DishSponge	kitchens,bathrooms
Drawer	kitchens,livingrooms,bedrooms,bathrooms
Dresser	livingrooms,bedrooms,bathrooms
Egg	kitchens
Faucet	kitchens,bathrooms
FloorLamp	livingrooms,bedrooms
Footstool	bedrooms
Fork	kitchens
Fridge	kitchens
GarbageCan	kitchens,livingrooms,bedrooms,bathrooms
HandTowel	bathrooms
HandTowelHolder	bathrooms
HousePlant	kitchens,livingrooms,bedrooms,bathrooms
Kettle	kitchens
KeyChain	livingrooms,bedrooms
Knife	kitchens
Ladle	kitchens
Laptop	kitchens,livingrooms,bedrooms
LaundryHamper	bedrooms
LaundryHamperLid	bedrooms
Lettuce	kitchens
LightSwitch	kitchens,livingrooms,bedrooms,bathrooms
Microwave	kitchens
Mirror	kitchens,livingrooms,bedrooms,bathrooms
Mug	kitchens,bedrooms
Newspaper	livingrooms
Ottoman	livingrooms,bedrooms
Painting	kitchens,livingrooms,bedrooms,bathrooms
Pan	kitchens
PaperTowel	kitchens,bathrooms
Pen	kitchens,livingrooms,bedrooms
Pencil	kitchens,livingrooms,bedrooms
PepperShaker	kitchens
Pillow	livingrooms,bedrooms
Plate	kitchens,livingrooms
Plunger	bathrooms
Poster	bedrooms
Pot	kitchens
Potato	kitchens
RemoteControl	livingrooms,bedrooms
Safe	kitchens,livingrooms,bedrooms
SaltShaker	kitchens
ScrubBrush	bathrooms
Shelf	kitchens,livingrooms,bedrooms,bathrooms
ShowerCurtain	bathrooms
ShowerDoor	bathrooms
ShowerGlass	bathrooms
ShowerHead	bathrooms
SideTable	livingrooms,bedrooms
Sink	kitchens,bathrooms
SinkBasin	kitchens,bathrooms
SoapBar	bathrooms
SoapBottle	kitchens,bathrooms
Sofa	livingrooms,bedrooms
Spatula	kitchens
Spoon	kitchens
SprayBottle	bathrooms
Statue	kitchens,livingrooms,bedrooms
StoveBurner	kitchens
StoveKnob	kitchens
TeddyBear	bedrooms
Television	livingrooms,bedrooms
TennisRacket	bedrooms
TissueBox	livingrooms,bedrooms,bathrooms
Toaster	kitchens
Toilet	bathrooms
ToiletPaper	bathrooms
ToiletPaperHanger	bathrooms
Tomato	kitchens
Towel	bathrooms
TowelHolder	bathrooms
TVStand	livingrooms
Vase	kitchens,livingrooms,bedrooms
Watch	livingrooms,bedrooms
WateringCan	livingrooms
Window	kitchens,livingrooms,bedrooms,bathrooms
WineBottle	kitchens
"""

OBJECT_TYPE_TO_SCENE_TYPES = OrderedDict()
for ot_tab_scene_types in _object_type_and_location_tsv.split("\n"):
    if ot_tab_scene_types != "":
        ot, scene_types_csv = ot_tab_scene_types.split("\t")
        OBJECT_TYPE_TO_SCENE_TYPES[ot] = tuple(sorted(scene_types_csv.split(",")))

SCENE_TYPE_TO_OBJECT_TYPES: Dict[str, Set[str]] = OrderedDict(
    ((k, set()) for k in ORDERED_SCENE_TYPES)
)
for ot_tab_scene_types in _object_type_and_location_tsv.split("\n"):
    if ot_tab_scene_types != "":
        ot, scene_types_csv = ot_tab_scene_types.split("\t")
        for scene_type in scene_types_csv.split(","):
            SCENE_TYPE_TO_OBJECT_TYPES[scene_type].add(ot)
