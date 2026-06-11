#!/usr/bin/env python3
"""
Combined DEEPX worker — loads SCDepthV3 + YOLO26m-cls in one process
to avoid concurrent DXRT device access from two processes.

Protocol (binary):
  main → worker : 1 byte  (0x01 = run depth, 0x02 = run cls, 0x00 = quit)
  worker → main : 1 byte 0x01 (done)
                  float  inf_ms
  Side effects:
    depth: writes /tmp/dd_depth_pane.bin  (960×1080×3)
    cls:   writes /tmp/dd_cls_pane.bin    (960×1080×3)
"""
import os, signal, struct, sys, time
import numpy as np
import cv2

# Suppress DXRT noise on fd-1
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_pipe_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

from dx_engine import InferenceEngine, InferenceOption

DEPTH_MODEL      = '/data/local/tmp/scdepthv3.dxnn'
CLS_MODEL        = '/data/local/tmp/yolo26m-cls.dxnn'
FRAME0_FILE      = '/tmp/dd_frame0.bin'
FRAME1_FILE      = '/tmp/dd_frame1.bin'
DEPTH_PANE_FILE  = '/tmp/dd_depth_pane.bin'
CLS_PANE_FILE    = '/tmp/dd_cls_pane.bin'
CAM_W, CAM_H     = 1280, 720
DEPTH_W, DEPTH_H = 320, 256
CLS_SIZE         = 224
PANE_W, PANE_H   = 960, 1080
TOP_K            = 5

# ImageNet-1k labels (compact)
IMAGENET_LABELS = (
    'tench','goldfish','great white shark','tiger shark','hammerhead',
    'electric ray','stingray','cock','hen','ostrich','brambling','goldfinch',
    'house finch','junco','indigo bunting','American robin','bulbul','jay',
    'magpie','chickadee','American dipper','kite','bald eagle','vulture',
    'great grey owl','fire salamander','smooth newt','eft','spotted salamander',
    'axolotl','American bullfrog','tree frog','tailed frog','loggerhead',
    'leatherback turtle','mud turtle','terrapin','box turtle','banded gecko',
    'green iguana','Carolina anole','desert grassland whiptail lizard',
    'agama','frilled-neck lizard','alligator lizard','Gila monster',
    'European green lizard','chameleon','Komodo dragon','Nile crocodile',
    'American alligator','triceratops','worm snake','ring-necked snake',
    'eastern hog-nosed snake','smooth green snake','kingsnake','garter snake',
    'water snake','vine snake','night snake','boa constrictor','African rock python',
    'Indian cobra','green mamba','sea snake','Saharan horned viper',
    'eastern diamondback rattlesnake','sidewinder','trilobite','harvestman',
    'scorpion','yellow garden spider','barn spider','European garden spider',
    'southern black widow','tarantula','wolf spider','tick','centipede',
    'black grouse','ptarmigan','ruffed grouse','prairie grouse','peacock',
    'quail','partridge','grey parrot','macaw','sulphur-crested cockatoo',
    'lorikeet','coucal','bee eater','hornbill','hummingbird','jacamar',
    'toucan','duck','red-breasted merganser','goose','black swan','tusker',
    'echidna','platypus','wallaby','koala','wombat','jellyfish','sea anemone',
    'brain coral','flatworm','nematode','conch','snail','slug','sea slug',
    'chiton','chambered nautilus','Dungeness crab','rock crab','fiddler crab',
    'red king crab','American lobster','spiny lobster','crayfish','hermit crab',
    'isopod','white stork','black stork','spoonbill','flamingo',
    'little blue heron','great egret','bittern','crane','limpkin',
    'common gallinule','American coot','bustard','ruddy turnstone',
    'dunlin','common redshank','dowitcher','oystercatcher','pelican',
    'king penguin','albatross','grey whale','killer whale','dugong',
    'sea lion','Chihuahua','Japanese Chin','Maltese','Pekingese','Shih Tzu',
    'King Charles Spaniel','Papillon','toy terrier','Rhodesian Ridgeback',
    'Afghan Hound','Basset Hound','Beagle','Bloodhound','Bluetick Coonhound',
    'Black and Tan Coonhound','Treeing Walker Coonhound','English foxhound',
    'Redbone Coonhound','borzoi','Irish Wolfhound','Italian Greyhound',
    'Whippet','Ibizan Hound','Norwegian Elkhound','Otterhound','Saluki',
    'Scottish Deerhound','Weimaraner','Staffordshire Bull Terrier',
    'American Staffordshire Terrier','Bedlington Terrier','Border Terrier',
    'Kerry Blue Terrier','Irish Terrier','Norfolk Terrier','Norwich Terrier',
    'Yorkshire Terrier','Wire Fox Terrier','Lakeland Terrier','Sealyham Terrier',
    'Airedale Terrier','Cairn Terrier','Australian Terrier','Dandie Dinmont Terrier',
    'Boston Terrier','Miniature Schnauzer','Giant Schnauzer','Standard Schnauzer',
    'Scottish Terrier','Tibetan Terrier','Australian Silky Terrier',
    'Soft-coated Wheaten Terrier','West Highland White Terrier','Lhasa Apso',
    'Flat-Coated Retriever','Curly-coated Retriever','Golden Retriever',
    'Labrador Retriever','Chesapeake Bay Retriever','German Shorthaired Pointer',
    'Vizsla','English Setter','Irish Setter','Gordon Setter','Brittany',
    'Clumber Spaniel','English Springer Spaniel','Welsh Springer Spaniel',
    'Cocker Spaniels','Sussex Spaniel','Irish Water Spaniel','Kuvasz',
    'Schipperke','Groenendael','Malinois','Briard','Australian Kelpie',
    'Komondor','Old English Sheepdog','Shetland Sheepdog','collie',
    'Border Collie','Bouvier des Flandres','Rottweiler','German Shepherd Dog',
    'Dobermann','Miniature Pinscher','Greater Swiss Mountain Dog','Bernese Mountain Dog',
    'Appenzeller Sennenhund','Entlebucher Sennenhund','Boxer','Bullmastiff',
    'Tibetan Mastiff','French Bulldog','Great Dane','St. Bernard','husky',
    'Alaskan Malamute','Siberian Husky','Dalmatian','Affenpinscher','Basenji',
    'pug','Leonberger','Newfoundland dog','Pyrenean Mountain Dog','Samoyed',
    'Pomeranian','Chow Chow','Keeshond','Brussels Griffon','Pembroke Welsh Corgi',
    'Cardigan Welsh Corgi','Toy Poodle','Miniature Poodle','Standard Poodle',
    'Mexican hairless dog','grey wolf','Alaskan tundra wolf','red wolf',
    'coyote','dingo','dhole','African wild dog','hyena','red fox','kit fox',
    'Arctic fox','grey fox','tabby cat','tiger cat','Persian cat','Siamese cat',
    'Egyptian Mau','cougar','lynx','leopard','snow leopard','jaguar','lion',
    'tiger','cheetah','brown bear','American black bear','polar bear',
    'sloth bear','mongoose','meerkat','tiger beetle','ladybug','ground beetle',
    'longhorn beetle','leaf beetle','dung beetle','rhinoceros beetle',
    'weevil','fly','bee','ant','grasshopper','cricket','stick insect',
    'cockroach','mantis','cicada','leafhopper','lacewing','dragonfly',
    'damselfly','cabbage white butterfly','monarch butterfly','small white',
    'sulphur butterfly','zebra longwing','sea cucumber','rabbit','hamster',
    'squirrel','marmot','beaver','guinea pig','common sorrel','zebra',
    'pig','wild boar','warthog','hippopotamus','ox','water buffalo','bison',
    'ram','bighorn sheep','Alpine ibex','hartebeest','impala','gazelle',
    'dromedary','llama','weasel','mink','European polecat','black-footed ferret',
    'otter','skunk','badger','armadillo','three-toed sloth','orangutan',
    'gorilla','chimpanzee','gibbon','siamang','guenon','patas monkey',
    'baboon','macaque','langur','black-and-white colobus','proboscis monkey',
    'marmoset','white-headed capuchin','howler monkey','titi monkey',
    'Geoffroys spider monkey','common squirrel monkey','ring-tailed lemur',
    'indri','Asian elephant','African bush elephant','red panda','giant panda',
    'snoek','eel','coho salmon','rock beauty','clownfish','sturgeon',
    'gar','lionfish','puffer fish','abacus','abaya','academic gown','accordion',
    'acoustic guitar','aircraft carrier','airliner','airship','altar',
    'ambulance','amphibious vehicle','analog clock','apiary','apron',
    'trash can','assault rifle','backpack','bakery','balance beam','balloon',
    'ballpoint pen','Band-Aid','banjo','baluster','barbershop','barn',
    'barometer','barrel','wheelbarrow','baseball','basketball','bassinet',
    'bassoon','swimcap','bath towel','bathtub','station wagon','lighthouse',
    'beaker','military hat','beer bottle','beer glass','bell tower','baby bib',
    'tandem bicycle','bikini','ring binder','binoculars','birdhouse',
    'boathouse','bobsleigh','flask','bottlecap','hunting bow','bow tie',
    'brass memorial plaque','bra','breakwater','breastplate','broom',
    'bucket','buckle','bulletproof vest','high-speed train','butcher shop',
    'taxicab','cauldron','candle','cannon','canoe','can opener','cardigan',
    'car mirror','carousel','toolbox','cello','mobile phone','chain',
    'chain-link fence','chain mail','chainsaw','chest','chiffonier','chime',
    'china cabinet','Christmas stocking','church','movie theater','cleaver',
    'cliff dwelling','cloak','clogs','cocktail shaker','coffee mug',
    'coffeemaker','spiral galaxy','combination lock','computer keyboard',
    'candy store','container ship','convertible','corkscrew','cornet',
    'cowboy boot','cowboy hat','cradle','construction crane','crash helmet',
    'crate','infant bed','Crock Pot','croquet ball','crutch','cuirass',
    'dam','desk','desktop computer','rotary dial telephone','diaper',
    'digital clock','digital watch','dining table','dishcloth','dishwasher',
    'disc brake','dock','dog sled','dome','doormat','drilling rig','drum',
    'drumstick','dumbbell','Dutch oven','electric fan','electric guitar',
    'electric locomotive','entertainment center','envelope','espresso machine',
    'face powder','feather boa','filing cabinet','fireboat','police van',
    'fire truck','ambulance','flagpole','flute','folding chair',
    'football helmet','forklift','fountain','fountain pen','four-poster bed',
    'freight car','French horn','frying pan','fur coat','garbage truck',
    'gas mask','gas pump','goblet','go-kart','golf ball','golf cart',
    'gondola','gong','gown','grand piano','greenhouse','radiator grille',
    'grocery store','guillotine','hair clip','hair spray','half-track',
    'hammer','hamper','hair dryer','hand-held computer','handkerchief',
    'hard disk drive','harmonica','harp','combine harvester','hatchet',
    'holster','home theater','honeycomb','hook','hoop skirt',
    'gymnastic horizontal bar','horse-drawn vehicle','hourglass','iPod',
    'clothes iron','carved pumpkin','jeans','jeep','T-shirt','jigsaw puzzle',
    'pulled rickshaw','joystick','kimono','knee pad','knot','lab coat',
    'ladle','lampshade','laptop computer','lawn mower','lens cap',
    'letter opener','library','lifeboat','lighter','limousine','ocean liner',
    'lipstick','slip-on shoe','lotion','music speaker','magnetic compass',
    'mailbag','mailbox','tights','one-piece bathing suit','manhole cover',
    'maraca','marimba','mask','matchstick','maypole','maze','measuring cup',
    'medicine cabinet','megalith','microphone','microwave oven',
    'military uniform','milk can','minibus','miniskirt','minivan','missile',
    'mitten','mixing bowl','mobile home','Model T','modem','monastery',
    'monitor','moped','mortar','square academic cap','mosque','mosquito net',
    'vespa','mountain bike','tent','computer mouse','mousetrap','moving van',
    'muzzle','metal nail','neck brace','necklace','baby pacifier',
    'notebook computer','obelisk','oboe','ocarina','odometer','oil filter',
    'pipe organ','oscilloscope','overskirt','bullock cart','oxygen mask',
    'product packet','paddle','paddle wheel','padlock','paintbrush','pajamas',
    'palace','pan flute','paper towel','parachute','parallel bars',
    'park bench','parking meter','railroad car','patio','payphone',
    'pedestal','pencil case','pencil sharpener','perfume','Petri dish',
    'photocopier','plectrum','Pickelhaube','picket fence','pickup truck',
    'pier','piggy bank','pill bottle','pillow','ping-pong ball','pinwheel',
    'pirate ship','pitcher','hand plane','planetarium','plastic bag',
    'plate rack','farm plow','plunger','Polaroid camera','pole',
    'police uniform','poncho','pool table','soda bottle','plant pot',
    "potter's wheel",'power drill','prayer rug','printer','prison',
    'missile','projectile','projector','hockey puck','punching bag','purse',
    'quill','racing car','racket','radio','radio telescope','rain barrel',
    'recreational vehicle','fishing casting reel','reflex camera',
    'refrigerator','remote control','restaurant','revolver','rifle',
    'rocking chair','rotisserie','eraser','rugby ball','ruler measuring stick',
    'sneaker','safe','safety pin','salt shaker','sandal','sarong','saxophone',
    'scabbard','weighing scale','school bus','schooner','scoreboard',
    'sewing machine','shield','shoe store','shoji screen','shopping basket',
    'shopping cart','shovel','shower cap','shower curtain','ski','balaclava',
    'sleeping bag','slide rule','sliding door','slot machine','snorkel',
    'snowmobile','snowplow','soap dispenser','soccer ball','sock',
    'solar thermal collector','sombrero','soup bowl','keyboard space bar',
    'space heater','space shuttle','spatula','motorboat','spider web',
    'spindle','sports car','spotlight','stage','steam locomotive',
    'through arch bridge','steel drum','stethoscope','scarf','stone wall',
    'stopwatch','stove','strainer','tram','stretcher','couch','stupa',
    'submarine','suit','sundial','sunglass','sunglasses','sunscreen',
    'suspension bridge','mop','sweatshirt','swim trunks','swing','switch',
    'syringe','table lamp','tank','tape player','teapot','teddy bear',
    'television','tennis ball','thatched roof','front curtain','thimble',
    'threshing machine','throne','tile roof','toaster','tobacco shop',
    'toilet seat','torch','totem pole','tow truck','toy store','tractor',
    'semi-trailer truck','tray','trench coat','tricycle','trimaran',
    'tripod','triumphal arch','trolleybus','trombone','hot tub','turnstile',
    'typewriter keyboard','umbrella','unicycle','upright piano',
    'vacuum cleaner','vase','viaduct','violin','volleyball','waffle iron',
    'wall clock','wallet','wardrobe','military aircraft','sink',
    'washing machine','water bottle','water jug','water tower','whiskey jug',
    'whistle','hair wig','window screen','window shade','Windsor tie',
    'wine bottle','wing','wok','wooden spoon','wool','split-rail fence',
    'shipwreck','sailboat','yurt','website','comic book','crossword',
    'traffic sign','traffic light','dust jacket','menu','plate','guacamole',
    'consomme','hot pot','trifle','ice cream','popsicle','baguette','bagel',
    'pretzel','cheeseburger','hot dog','mashed potato','cabbage','broccoli',
    'cauliflower','zucchini','spaghetti squash','acorn squash',
    'butternut squash','cucumber','artichoke','bell pepper','mushroom',
    'Granny Smith apple','strawberry','orange','lemon','fig','pineapple',
    'banana','jackfruit','custard apple','pomegranate','hay','carbonara',
    'chocolate syrup','dough','meatloaf','pizza','pot pie','burrito',
    'red wine','espresso','tea cup','eggnog','alp','bubble','cliff',
    'coral reef','geyser','lakeshore','promontory','sandbar','beach',
    'valley','volcano','baseball player','bridegroom','scuba diver',
    'rapeseed','daisy','yellow ladys slipper','corn','acorn','rose hip',
    'horse chestnut seed','coral fungus','agaric','gyromitra',
    'stinkhorn mushroom','earth star','hen of the woods','bolete',
    'ear of corn','toilet paper',
)

# ── init both engines sequentially ────────────────────────────────────────────
opt = InferenceOption()
try:
    opt.set_buffer_count(1)
except Exception:
    pass

depth_engine = cls_engine = None
init_errors = []

try:
    depth_engine = InferenceEngine(DEPTH_MODEL, opt)
    sys.stderr.write('[dual_deepx] SCDepthV3 loaded\n')
    sys.stderr.flush()
except Exception as e:
    init_errors.append(f'SCDepthV3: {e}')

try:
    cls_engine = InferenceEngine(CLS_MODEL, opt)
    sys.stderr.write('[dual_deepx] YOLO26m-cls loaded\n')
    sys.stderr.flush()
except Exception as e:
    init_errors.append(f'YOLO26m-cls: {e}')

for err in init_errors:
    sys.stderr.write(f'[dual_deepx] init failed: {err}\n')
    sys.stderr.flush()

_pipe_out.write(b'READY\n')

# ── pre-allocate ───────────────────────────────────────────────────────────────
_depth_inp = np.empty((DEPTH_H, DEPTH_W, 3), np.uint8)
_cls_inp   = np.empty((CLS_SIZE, CLS_SIZE, 3), np.uint8)
_pane_buf  = np.empty((PANE_H, PANE_W, 3), np.uint8)

def center_crop_resize(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    cv2.resize(img[y0:y0+s, x0:x0+s], (size, size),
               dst=_cls_inp, interpolation=cv2.INTER_AREA)
    return _cls_inp

def softmax(x):
    ex = np.exp(x - x.max())
    return ex / ex.sum()

# ── main loop ─────────────────────────────────────────────────────────────────
stdin_fd = sys.stdin.buffer
while True:
    cmd = stdin_fd.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)

    # ── Depth inference ───────────────────────────────────────────────────────
    if cmd == b'\x01':
        try:
            raw = np.fromfile(FRAME0_FILE, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
        except Exception:
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(DEPTH_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))
            continue

        try:
            cv2.resize(raw, (DEPTH_W, DEPTH_H), dst=_depth_inp, interpolation=cv2.INTER_AREA)
            t0 = time.time()
            outs = depth_engine.run(np.expand_dims(_depth_inp, 0))
            inf_ms = (time.time() - t0) * 1000.0

            depth = outs[0].reshape(DEPTH_H, DEPTH_W).astype(np.float32)
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_u8 = np.zeros((DEPTH_H, DEPTH_W), np.uint8)

            colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            cv2.resize(colored, (PANE_W, PANE_H), dst=_pane_buf, interpolation=cv2.INTER_LINEAR)
            cv2.putText(_pane_buf, 'SCDepthV3  (DEEPX)', (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(_pane_buf, f'cam0  {inf_ms:.0f}ms', (16, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
            _pane_buf.tofile(DEPTH_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', float(inf_ms)))
        except Exception as e:
            sys.stderr.write(f'[dual_deepx] depth error: {e}\n')
            sys.stderr.flush()
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(DEPTH_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))

    # ── Classification inference ───────────────────────────────────────────────
    elif cmd == b'\x02':
        try:
            raw = np.fromfile(FRAME1_FILE, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
        except Exception:
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(CLS_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))
            continue

        try:
            inp = center_crop_resize(raw, CLS_SIZE)
            t0 = time.time()
            outs = cls_engine.run(np.expand_dims(inp, 0))
            inf_ms = (time.time() - t0) * 1000.0

            probs = softmax(outs[0].reshape(-1).astype(np.float32))
            top5  = np.argsort(probs)[::-1][:TOP_K]

            cv2.resize(raw, (PANE_W, PANE_H), dst=_pane_buf, interpolation=cv2.INTER_AREA)
            cv2.putText(_pane_buf, 'YOLO26m-cls  (DEEPX)', (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(_pane_buf, f'cam1  {inf_ms:.0f}ms', (16, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

            bar_y = PANE_H - 300
            cv2.rectangle(_pane_buf, (0, bar_y), (PANE_W, PANE_H), (0, 0, 0), -1)
            cv2.putText(_pane_buf, 'Top-5 Classification:', (16, bar_y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 1, cv2.LINE_AA)
            n_labels = len(IMAGENET_LABELS)
            for rank, idx in enumerate(top5):
                label = IMAGENET_LABELS[idx] if idx < n_labels else f'class_{idx}'
                score = float(probs[idx])
                bw = int(score * (PANE_W - 32))
                y = bar_y + 60 + rank * 48
                cv2.rectangle(_pane_buf, (16, y), (16 + bw, y + 32), (30, 100, 200), -1)
                cv2.putText(_pane_buf, f'{score:.2%}  {label}', (20, y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

            _pane_buf.tofile(CLS_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', float(inf_ms)))
        except Exception as e:
            sys.stderr.write(f'[dual_deepx] cls error: {e}\n')
            sys.stderr.flush()
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(CLS_PANE_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))
