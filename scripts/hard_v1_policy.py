"""hard-v1 family `long_policy`: a generated policy document (travel, home contents, employee health benefits or device
protection) of roughly 800-5,000 tokens with numbered sections, definitions, limits, sublimits, a deductible, waiting
periods, a notice condition, exclusions with exceptions and boilerplate general conditions, followed by a claim. Two
questions: the settlement outcome (choice among five, e.g. decline under a named exclusion, pay subject to a sublimit, pay
the loss less the deductible) and the amount payable (choice among five computed amounts).

The label is `solve(facts)`: a small rule engine over the policy parameters and the claim facts stored in `_meta.facts`.
The generator picks a target outcome, builds a claim for it, and keeps the record only if the engine agrees and exactly
one reason to decline applies (so there is never a precedence question). Distractors are built in on purpose: an exclusion
whose condition holds but whose exception also holds, a threshold exclusion missed by a small margin, a sublimit for a
class the item does not belong to, and sections for covers the claim does not touch.
"""
import re
from datetime import date, timedelta

from scripts.hard_v1_common import choice_q, day, dollars, money, person, roman, value_q

SRC = "hard_long_policy"

# ---------------------------------------------------------------------------------------------------------------- domains
# Text uses voice tokens ({we}, {We}, {our}, {us}, {you}, {You}, {your}, {Your}) that a template renders in the first or
# the third person; only modal verbs follow {we}/{you} so both voices stay grammatical. Case sentences use {first},
# {item} (the item, with "the") and {a_item}.

UNATTENDED = {"key": "unattended", "title": "Unattended property", "kind": "flag",
              "text": "loss or theft of property left unattended in a vehicle or in a public place",
              "exception": "This exclusion does not apply to property locked out of sight in the boot or trunk of a locked vehicle",
              "yes": ["{First} had left {item} in a parked car while visiting a museum.",
                      "{item_cap} had been left in a hire car parked outside a restaurant during dinner."],
              "no": [],
              "exc_yes": ["The car was locked and {item} was in the closed boot, out of sight.",
                          "The vehicle was locked, and {item} was stowed in the covered trunk where it could not be seen."],
              "exc_no": ["The car was locked, but {item} was on the back seat in plain view.", "The car had been left unlocked."]}

DOMAINS = {
    "travel": {
        "titles": ["Travel Insurance Policy Wording", "Single-Trip Travel Protection Plan", "Annual Multi-Trip Travel Cover"],
        "issuers": ["Meridian Travel Insurance Ltd", "Bluewater Assurance", "Compass Mutual", "Northstar Underwriting"],
        "insurer": "Insurer", "holder": "Insured Person", "fee": "excess", "event": "incident",
        "extension": "Adventure Sports extension",
        "details": ["The trip was booked through an online travel agent.", "{First} was travelling with two colleagues.",
                    "The return flight was scheduled for the following week.", "{First} paid for the trip with a credit card.",
                    "The destination was a coastal town {first} had visited before."],
        "categories": {
            "baggage": {
                "title": "Baggage and Personal Belongings", "limit": [1500, 2000, 3000], "fee": [50, 75, 100], "waiting": [0],
                "cover": "{We} will pay for the loss of, theft of, or damage to {your} baggage and personal belongings during the trip.",
                "classes": {"valuables": ("Valuables", "laptops, tablets, mobile phones, cameras, watches and jewellery", [250, 400, 500]),
                            "sports": ("Sports equipment", "skis, snowboards, golf clubs, surfboards and bicycles", [300, 450])},
                "items": [("a laptop", "the laptop", "valuables"), ("a camera", "the camera", "valuables"),
                          ("a wristwatch", "the wristwatch", "valuables"), ("a set of golf clubs", "the golf clubs", "sports"),
                          ("a suitcase of clothes", "the suitcase", None), ("a pair of prescription glasses", "the glasses", None),
                          ("a travel backpack and its contents", "the backpack", None)],
                "benign": ["{item_cap} was stolen from {first}'s locked hotel room.", "The airline lost {item} on the outbound flight."],
                "exclusions": [UNATTENDED,
                               {"key": "confiscation", "title": "Confiscation", "kind": "flag",
                                "text": "loss caused by confiscation or detention by customs or other officials",
                                "yes": ["Customs officers at the border confiscated {item}."],
                                "no": ["{item_cap} passed through customs without being inspected."]},
                               {"key": "wear", "title": "Wear and tear", "kind": "flag",
                                "text": "damage caused by wear and tear or gradual deterioration",
                                "yes": ["The damage to {item} was caused by years of gradual wear."],
                                "no": []}]},
            "medical": {
                "title": "Emergency Medical Expenses", "limit": [10000, 25000, 50000], "fee": [100, 150, 250], "waiting": [0],
                "cover": "{We} will pay reasonable and necessary emergency medical costs incurred during the trip.",
                "classes": {"dental": ("Dental treatment", "treatment of the teeth or gums", [250, 400]),
                            "physio": ("Physiotherapy", "treatment of muscles and joints by a physiotherapist", [300, 500])},
                "items": [("hospital treatment for a broken ankle", "the ankle treatment", None),
                          ("emergency treatment for a cracked molar", "the molar treatment", "dental"),
                          ("six physiotherapy sessions for a torn ligament", "the physiotherapy", "physio"),
                          ("a hospital stay for acute appendicitis", "the hospital stay", None)],
                "benign": ["{First} fell ill suddenly on the third day of the trip.", "The treating doctor confirmed the treatment could not wait until {first} returned home."],
                "exclusions": [
                    {"key": "pre_existing", "title": "Pre-existing conditions", "kind": "threshold", "cmp": "le", "thr": [3, 6, 12], "gap": 1,
                     "term": "Pre-existing condition", "definition": "any illness or injury for which {you} received treatment or advice in the {thr} months before the policy start date",
                     "text": "treatment of a pre-existing condition",
                     "value": "{First}'s records show treatment for the same condition {v} months before the policy start date."},
                    {"key": "hazardous", "title": "Hazardous activities", "kind": "threshold", "cmp": "gt", "thr": [3000, 4000, 4500], "gap": 150, "unit": ",",
                     "term": "Hazardous activity", "definition": "skydiving, motor racing, or trekking or climbing above {thr} metres",
                     "text": "injury or illness arising from a hazardous activity",
                     "exception": "This exclusion does not apply if the {extension} is shown in {your} schedule",
                     "value": "The problem began while {first} was trekking at an altitude of {v} metres.",
                     "exc_yes": ["The schedule shows that the {extension} was purchased."],
                     "exc_no": ["No extensions are shown in the schedule."]},
                    {"key": "intoxication", "title": "Alcohol and drugs", "kind": "flag",
                     "text": "any claim arising from {your} being under the influence of alcohol or drugs",
                     "yes": ["The hospital report records that {first} was heavily intoxicated when the problem began."],
                     "no": ["The hospital report notes that {first} had not been drinking."]}]},
            "cancellation": {
                "title": "Trip Cancellation", "limit": [2000, 3000, 5000], "fee": [50, 100], "waiting": [7, 10, 14],
                "cover": "{We} will pay non-refundable travel and accommodation costs if {you} must cancel the trip for a reason beyond {your} control.",
                "classes": {"excursions": ("Excursions and tickets", "pre-booked excursions, tours, and event or theatre tickets", [200, 300])},
                "items": [("non-refundable flights and hotel bookings", "the flights and hotel", None),
                          ("a non-refundable guided tour package", "the tour package", None),
                          ("pre-booked concert and excursion tickets", "the tickets", "excursions")],
                "benign": ["The trip was cancelled because {first} was admitted to hospital a week before departure.",
                           "{First} cancelled after receiving a jury summons for the travel dates."],
                "exclusions": [
                    {"key": "known_event", "title": "Known events", "kind": "flag",
                     "text": "cancellation caused by a strike or other event that had been publicly announced before {you} bought the policy",
                     "yes": ["The cancellation followed an air traffic control strike that had been announced before {first} bought the policy."],
                     "no": []},
                    {"key": "change_of_mind", "title": "Disinclination to travel", "kind": "flag",
                     "text": "cancellation because {you} no longer wish to travel",
                     "yes": ["{First} decided not to travel after finding a cheaper holiday elsewhere."],
                     "no": []}]},
            "delay": {
                "title": "Travel Delay", "limit": [300, 500, 750], "fee": [25, 50], "waiting": [0], "optional": True,
                "cover": "{We} will pay reasonable costs of meals and accommodation if {your} outbound departure is delayed.",
                "classes": {},
                "items": [("meals and a hotel night during a departure delay", "the meals and hotel", None),
                          ("meals and taxis during a long airport delay", "the meals and taxis", None)],
                "benign": ["The delay was caused by a technical fault with the aircraft."],
                "exclusions": [
                    {"key": "short_delay", "title": "Short delays", "kind": "threshold", "cmp": "le", "thr": [6, 8, 12], "gap": 1,
                     "term": "Qualifying delay", "definition": "a delay to the scheduled departure of more than {thr} hours",
                     "text": "any delay that is not a qualifying delay",
                     "value": "The departure was delayed by {v} hours."}]},
        }},
    "home": {
        "titles": ["Home Contents Insurance Policy", "Household Contents Cover: Policy Booklet", "Renters Contents Protection"],
        "issuers": ["Hearthstone Insurance plc", "Keystone Home Mutual", "Lantern General Insurance", "Oakfield Assurance"],
        "insurer": "Insurer", "holder": "Policyholder", "fee": "excess", "event": "incident",
        "extension": "Extended Theft endorsement",
        "details": ["{First} has lived at the property for four years.", "The building is a converted warehouse with six flats.",
                    "{First} works from home three days a week.", "A neighbour reported the incident to the building manager."],
        "categories": {
            "theft": {
                "title": "Theft", "limit": [10000, 20000, 40000], "fee": [100, 250, 500], "waiting": [0],
                "cover": "{We} will pay for contents stolen from {your} home.",
                "classes": {"high_risk": ("High-risk items", "jewellery, watches, laptops, games consoles and cameras", [1000, 1500, 2500]),
                            "bicycles": ("Bicycles", "bicycles and electric scooters", [500, 800])},
                "items": [("an engagement ring", "the ring", "high_risk"), ("a laptop", "the laptop", "high_risk"),
                          ("a games console", "the games console", "high_risk"), ("a road bicycle", "the bicycle", "bicycles"),
                          ("a television", "the television", None), ("a set of power tools", "the power tools", None),
                          ("a sofa and armchair", "the sofa and armchair", None)],
                "benign": ["Burglars forced the back door and took {item}."],
                "exclusions": [
                    {"key": "unoccupied", "title": "Unoccupied home", "kind": "threshold", "cmp": "gt", "thr": [30, 45, 60], "gap": 2,
                     "term": "Unoccupied", "definition": "not lived in by {you} or a member of {your} family for more than {thr} consecutive days",
                     "text": "theft while the home is unoccupied",
                     "value": "The home had been empty for {v} consecutive days when the theft happened."},
                    {"key": "no_forced_entry", "title": "Theft without forced entry", "kind": "flag",
                     "text": "theft where there is no sign of forced entry to the home",
                     "exception": "This exclusion does not apply where the thief gained entry by deception, for example by posing as a utility worker",
                     "yes": ["Police found no sign of forced entry."],
                     "exc_yes": ["A man posing as a gas engineer was let in by {first} and took {item} while {first} was in another room."],
                     "exc_no": ["The police report says a window had been left open."],
                     "no": ["The police report describes a broken lock on the front door."]},
                    {"key": "outbuilding", "title": "Outbuildings", "kind": "flag",
                     "text": "theft from sheds, garages or other outbuildings",
                     "exception": "This exclusion does not apply if the outbuilding was locked and the theft involved forcible entry",
                     "yes": ["{item_cap} was taken from the garden shed."],
                     "exc_yes": ["The shed was padlocked and the thieves cut through the padlock."],
                     "exc_no": ["The shed door had been left unlocked."],
                     "no": []}]},
            "water": {
                "title": "Escape of Water", "limit": [15000, 30000], "fee": [250, 500, 1000], "waiting": [0],
                "cover": "{We} will pay for contents damaged by water escaping from a fixed pipe, tank or domestic appliance.",
                "classes": {"art": ("Art and antiques", "paintings, antiques, rugs and other works of art", [1000, 2000])},
                "items": [("flooring and furniture", "the flooring and furniture", None), ("an antique rug", "the antique rug", "art"),
                          ("a framed oil painting", "the painting", "art"), ("a bed and wardrobe", "the bed and wardrobe", None)],
                "benign": ["A pipe under the kitchen sink burst suddenly during a cold night and soaked {item}."],
                "exclusions": [
                    {"key": "gradual", "title": "Gradual leaks", "kind": "flag",
                     "text": "damage caused by water leaking gradually over a period of time",
                     "yes": ["The plumber's report says the pipe had been leaking slowly for several months before {item} was damaged."],
                     "no": ["The plumber's report says the pipe failed suddenly."]},
                    {"key": "sealant", "title": "Failed sealant", "kind": "flag",
                     "text": "damage caused by the failure of grout or sealant around baths and showers",
                     "yes": ["The water came through failed grout around the shower tray."],
                     "no": []}]},
            "accidental": {
                "title": "Accidental Damage", "limit": [2000, 3000, 5000], "fee": [50, 100], "waiting": [0], "optional": True,
                "cover": "{We} will pay for sudden and unexpected accidental damage to {your} contents inside the home.",
                "classes": {"portable": ("Portable electronics", "laptops, tablets and mobile phones", [500, 750])},
                "items": [("a television", "the television", None), ("a laptop", "the laptop", "portable"),
                          ("a glass dining table", "the table", None), ("a tablet", "the tablet", "portable")],
                "benign": ["{First} knocked {item} over while moving furniture."],
                "exclusions": [
                    {"key": "pets", "title": "Pets", "kind": "flag", "text": "damage caused by pets",
                     "yes": ["{First}'s dog knocked {item} off its stand."], "no": []},
                    {"key": "wear_home", "title": "Wear and tear", "kind": "flag", "text": "wear and tear, or damage that happens gradually",
                     "yes": ["The damage developed gradually over several years of use."], "no": []}]},
            "flood": {
                "title": "Flood", "limit": [20000, 40000], "fee": [500, 1000], "waiting": [14, 30, 60],
                "cover": "{We} will pay for contents damaged by flood water entering {your} home.",
                "classes": {"garden": ("Garden items", "garden furniture, plants and outdoor tools", [500, 1000])},
                "items": [("furniture and appliances on the ground floor", "the furniture and appliances", None),
                          ("garden furniture and tools", "the garden furniture and tools", "garden")],
                "benign": ["Heavy rain caused the nearby river to burst its banks and flood the street."],
                "exclusions": [
                    {"key": "flood_warning", "title": "Flood warnings", "kind": "flag",
                     "text": "flood damage where a flood warning for {your} area had been issued before the policy started",
                     "yes": ["A flood warning for the area had been issued three days before the policy start date."],
                     "no": ["No flood warning had been issued before the policy started."]}]},
        }},
    "benefits": {
        "titles": ["Employee Health Benefits Plan: Summary Plan Description", "Staff Health Reimbursement Plan Rules", "Group Health Cash Plan"],
        "issuers": ["the Benefits Trust", "Wellspring Benefits Administration", "the Staff Welfare Fund", "Cedar Benefits Services"],
        "insurer": "Plan", "holder": "Member", "fee": "deductible", "event": "treatment",
        "extension": "Enhanced Care option",
        "details": ["{First} joined the company's finance team two years ago.", "{First} submitted receipts through the benefits portal.",
                    "{First}'s spouse is also enrolled in the plan.", "The provider issued an itemised invoice."],
        "categories": {
            "dental": {
                "title": "Dental Care", "limit": [1000, 1500, 2000], "fee": [50, 100], "waiting": [60, 90],
                "cover": "{We} will reimburse the cost of dental treatment provided by a licensed dentist.",
                "classes": {"major": ("Major restorative work", "crowns, bridges, root canal treatment and implants", [500, 750]),
                            "ortho": ("Orthodontics", "braces and clear aligners", [600, 800])},
                "items": [("a filling", "the filling", None), ("a crown", "the crown", "major"), ("root canal treatment", "the root canal treatment", "major"),
                          ("a dental implant", "the implant", "major"), ("a scale and polish", "the scale and polish", None),
                          ("a veneer on a front tooth", "the veneer", None)],
                "benign": ["The dentist recommended the treatment after a routine check-up."],
                "exclusions": [
                    {"key": "cosmetic", "title": "Cosmetic treatment", "kind": "flag", "only": ["the veneer"],
                     "text": "treatment that is mainly cosmetic, such as whitening or veneers",
                     "exception": "This exclusion does not apply to treatment needed to repair damage caused by an accident",
                     "yes": ["The veneer was fitted to improve the appearance of the tooth."],
                     "exc_yes": ["The veneer repaired a front tooth chipped in a cycling accident."],
                     "exc_no": ["The dentist confirms the tooth had not been injured."], "no": []},
                    {"key": "out_of_network", "title": "Out-of-network providers", "kind": "flag",
                     "text": "treatment by a provider outside the plan's network",
                     "exception": "This exclusion does not apply to emergency treatment for sudden severe pain",
                     "yes": ["The dentist is not part of the plan's network."],
                     "exc_yes": ["It was emergency treatment for sudden severe pain on a weekend."],
                     "exc_no": ["The appointment was booked three weeks in advance."],
                     "no": ["The dentist is a network provider."]}]},
            "vision": {
                "title": "Vision Care", "limit": [300, 400, 500], "fee": [25, 40], "waiting": [0],
                "cover": "{We} will reimburse eye tests and prescription eyewear.",
                "classes": {"frames": ("Frames", "spectacle frames bought without new lenses", [120, 150])},
                "items": [("prescription glasses", "the glasses", None), ("new spectacle frames without new lenses", "the frames", "frames"),
                          ("a year's supply of contact lenses", "the contact lenses", None), ("designer sunglasses", "the sunglasses", None)],
                "benign": ["The optician issued a new prescription after an eye test."],
                "exclusions": [
                    {"key": "non_prescription", "title": "Non-prescription eyewear", "kind": "flag", "only": ["the sunglasses"],
                     "text": "eyewear without prescription lenses",
                     "yes": ["The sunglasses have plain lenses with no prescription."], "no": []},
                    {"key": "frequency", "title": "Replacement frequency", "kind": "threshold", "cmp": "lt", "thr": [12, 24], "gap": 1,
                     "term": "Replacement period", "definition": "{thr} months from the date of {your} last eyewear claim",
                     "text": "a further eyewear claim within the replacement period",
                     "value": "{First}'s previous eyewear claim was {v} months before this one."}]},
            "therapy": {
                "title": "Therapy Services", "limit": [1000, 1500, 2000], "fee": [50, 75], "waiting": [0],
                "cover": "{We} will reimburse physiotherapy, osteopathy and counselling provided by a registered practitioner.",
                "classes": {"massage": ("Massage therapy", "sports massage, massage therapy and reflexology", [200, 300])},
                "items": [("a course of physiotherapy sessions", "the physiotherapy", None), ("counselling sessions with a registered psychologist", "the counselling", None),
                          ("a course of sports massage sessions", "the sports massage", "massage"), ("osteopathy sessions for back pain", "the osteopathy", None)],
                "benign": ["{First}'s doctor referred {first} for the sessions in writing."],
                "exclusions": [
                    {"key": "work_injury", "title": "Work injuries", "kind": "flag",
                     "text": "treatment of an injury covered by workers' compensation",
                     "yes": ["The injury happened at work and was accepted as a workers' compensation claim."],
                     "no": ["The injury happened while {first} was gardening at home."]},
                    {"key": "unregistered", "title": "Unregistered practitioners", "kind": "flag",
                     "text": "treatment by a practitioner who is not registered with the relevant professional body",
                     "yes": ["The practitioner's registration had lapsed before the sessions began."], "no": []}]},
            "hearing": {
                "title": "Hearing Care", "limit": [1000, 2000], "fee": [100, 150], "waiting": [0], "optional": True,
                "cover": "{We} will reimburse hearing tests and hearing aids.",
                "classes": {"accessories": ("Hearing accessories", "batteries, chargers and cleaning kits", [100, 150])},
                "items": [("a pair of hearing aids", "the hearing aids", None), ("hearing aid batteries and a charger", "the batteries and charger", "accessories")],
                "benign": ["An audiologist recommended the purchase after a hearing test."],
                "exclusions": [
                    {"key": "replacement", "title": "Early replacement", "kind": "threshold", "cmp": "lt", "thr": [24, 36], "gap": 2,
                     "term": "Hearing aid replacement period", "definition": "{thr} months from the purchase of hearing aids funded by the plan",
                     "text": "replacement hearing aids bought within the hearing aid replacement period",
                     "value": "{First}'s previous plan-funded hearing aids were bought {v} months before this claim."}]},
        }},
    "device": {
        "titles": ["Device Protection Plan Terms", "Gadget Cover: Terms and Conditions", "Electronics Care Plan"],
        "issuers": ["Voltguard Protection Ltd", "Circuit Care Services", "Brightline Device Cover", "Tessera Warranty Company"],
        "insurer": "Provider", "holder": "Customer", "fee": "service fee", "event": "incident",
        "extension": "Loss Plus add-on",
        "details": ["{First} bought the device from an authorised retailer.", "The device is registered in {first}'s name.",
                    "{First} uses the device mainly for work.", "The retailer's receipt was uploaded with the claim."],
        "categories": {
            "accidental_damage": {
                "title": "Accidental Damage", "limit": [800, 1200, 1500], "fee": [49, 79, 99], "waiting": [0],
                "cover": "{We} will repair or replace {your} device if it is accidentally damaged.",
                "classes": {"accessories": ("Accessories", "cases, chargers, earbuds and styluses", [50, 100]),
                            "wearables": ("Wearables", "smartwatches and fitness trackers", [250, 300])},
                "items": [("a smartphone", "the smartphone", None), ("a laptop", "the laptop", None),
                          ("a pair of wireless earbuds", "the earbuds", "accessories"), ("a smartwatch", "the smartwatch", "wearables"),
                          ("a tablet", "the tablet", None)],
                "benign": ["{item_cap} slipped from {first}'s hand onto a stone floor."],
                "exclusions": [
                    {"key": "cosmetic_damage", "title": "Cosmetic damage", "kind": "flag",
                     "text": "cosmetic damage, such as scratches or dents, that does not affect how the device works",
                     "yes": ["{item_cap} still works normally; the damage is limited to scratches on the casing."], "no": []},
                    {"key": "unauthorised_repair", "title": "Unauthorised repairs", "kind": "flag",
                     "text": "damage following a repair by anyone other than {us} or an authorised repairer",
                     "exception": "This exclusion does not apply if {we} approved the repair in writing beforehand",
                     "yes": ["{item_cap} had previously been opened by an independent repair shop."],
                     "exc_yes": ["The provider had approved that repair in writing before it was carried out."],
                     "exc_no": ["Nobody asked the provider to approve that repair."],
                     "no": ["{item_cap} has never been repaired."]}]},
            "breakdown": {
                "title": "Breakdown", "limit": [600, 1000, 1500], "fee": [25, 49], "waiting": [30, 45],
                "cover": "{We} will repair or replace {your} device if it stops working because of an electrical or mechanical fault.",
                "classes": {"battery": ("Battery replacement", "replacement batteries and battery repairs", [80, 120])},
                "items": [("a laptop that no longer powers on", "the laptop", None), ("a smartphone battery that no longer holds a charge", "the battery", "battery"),
                          ("a games console with a failed disc drive", "the games console", None)],
                "benign": ["The technician found a failed component on the main board."],
                "exclusions": [
                    {"key": "manufacturer_warranty", "title": "Manufacturer's warranty period", "kind": "threshold", "cmp": "lt", "thr": [12, 24], "gap": 1,
                     "term": "Manufacturer's warranty period", "definition": "the first {thr} months after the device was bought",
                     "text": "breakdowns during the manufacturer's warranty period",
                     "value": "The device was bought {v} months before it broke down."},
                    {"key": "liquid", "title": "Liquid damage", "kind": "flag", "text": "breakdown caused by liquid getting into the device",
                     "yes": ["The technician found signs of liquid inside the device."], "no": ["The technician found no sign of liquid inside the device."]}]},
            "theft_loss": {
                "title": "Theft and Loss", "limit": [800, 1200], "fee": [99, 149], "waiting": [0], "optional": True,
                "cover": "{We} will replace {your} device if it is stolen or lost.",
                "classes": {"accessories_tl": ("Accessories", "cases, chargers and earbuds", [50, 100])},
                "items": [("a smartphone", "the smartphone", None), ("a laptop", "the laptop", None), ("a pair of earbuds", "the earbuds", "accessories_tl")],
                "benign": ["{item_cap} was taken from {first}'s bag on a crowded train."],
                "exclusions": [UNATTENDED,
                               {"key": "no_police_report", "title": "Unreported theft", "kind": "flag",
                                "text": "theft that was not reported to the police",
                                "yes": ["{First} did not report the theft to the police."], "no": ["{First} reported the theft to the police the same day."]}]},
        }},
}

GENERAL_EXCLUSIONS = [
    "loss or damage caused by war, invasion, or civil war", "loss caused by nuclear radiation or radioactive contamination",
    "any deliberate or criminal act by {you}", "loss arising from a government order or sanction that prohibits {us} from paying",
    "any fraudulent claim or any claim supported by false documents", "loss of value or loss of use that does not involve physical loss or damage",
]
EXTRA_DEFINITIONS = [
    ("Schedule", "the document issued with this policy that shows {your} details, the covers {you} bought and any extensions"),
    ("Policy period", "twelve months from the policy start date shown in {your} schedule"),
    ("Assessed loss", "the cost of repair or replacement agreed by {us}, before any {fee}, limit or sublimit is applied"),
    ("Limit", "the most {we} will pay for all claims under a section arising from one {event}"),
    ("Sublimit", "a lower limit that applies to a particular class of property or treatment within a section"),
    ("Incident", "a single event, or a series of events arising from one cause"),
    ("Receipt", "a document from the seller or provider showing the date, the amount paid and what was supplied"),
    ("Family", "{your} spouse or partner and any children who live with {you}"),
    ("Premium", "the amount {you} must pay for this policy, including any taxes"),
    ("Business day", "Monday to Friday, excluding public holidays"),
    ("Proof of purchase", "an original receipt, invoice or bank statement showing the item, the price and the date"),
]
DOMAIN_DEFINITIONS = {
    "travel": [("Trip", "a journey that starts and ends in {your} home country during the policy period"),
               ("Territory", "the countries in which cover applies, as shown in the schedule"),
               ("Travelling companion", "a person who has booked to travel with {you} on the same trip"),
               ("Public transport", "any scheduled train, bus, coach, ferry or aircraft")],
    "home": [("Home", "the private residence at the address shown in the schedule"),
             ("Contents", "household goods and personal possessions that belong to {you} or {your} family"),
             ("Forcible entry", "entry to the home that leaves visible damage to doors, windows or locks"),
             ("Outbuilding", "a shed, garage or greenhouse within the boundaries of the home")],
    "benefits": [("Network provider", "a practitioner or clinic that has an agreement with {us} to provide treatment"),
                 ("Plan year", "the twelve months starting on {your} coverage start date"),
                 ("Dependant", "{your} spouse or partner, or a child under 21 who is enrolled in the plan"),
                 ("Registered practitioner", "a practitioner registered with the professional body for their discipline")],
    "device": [("Device", "the item shown in the schedule, bought new from an authorised retailer"),
               ("Authorised repairer", "a repairer approved in writing by {us}"),
               ("Replacement device", "a device of the same or similar specification, which may be refurbished"),
               ("Breakdown", "the failure of a part to work as the manufacturer intended, from an internal electrical or mechanical fault")],
}
# boilerplate general conditions: (title, clauses); they carry no fact the claim depends on
FILLER = [
    ("Cancellation", ["{You} can cancel this policy within 14 days of receiving it and receive a full refund, provided no claim has been made.",
                      "After the first 14 days {you} can cancel at any time; {we} will refund the premium for the unexpired period less an administration charge of {n25}.",
                      "{We} can cancel this policy by giving {you} 30 days' written notice to the last address {we} hold."]),
    ("Other insurance", ["If any loss covered by this policy is also covered by another policy, {we} will pay only {our} proportionate share.",
                         "{You} must tell {us} about any other insurance that covers the same loss when {you} make a claim."]),
    ("Subrogation", ["{We} will be entitled to take over and conduct in {your} name the defence or settlement of any claim.",
                     "{We} can take action in {your} name to recover any payment {we} make from anyone responsible for the loss."]),
    ("Fraud", ["If any claim is fraudulent or exaggerated in any respect, {we} will not pay the claim and all cover under this policy may be cancelled.",
               "{We} may share information with fraud prevention agencies and other insurers."]),
    ("Governing law", ["This policy is governed by the law of the jurisdiction in which it was issued, unless both parties agree otherwise in writing.",
                       "Any dispute will be subject to the exclusive jurisdiction of the courts of that jurisdiction."]),
    ("Complaints", ["If {you} have a complaint, {you} should first contact {our} customer relations team, quoting the policy number.",
                    "{We} will acknowledge a complaint within five working days and aim to resolve it within eight weeks.",
                    "If {you} remain dissatisfied, {you} may be able to refer the complaint to an independent ombudsman service."]),
    ("Data protection", ["{We} will use the personal information {you} provide to administer this policy and to handle claims.",
                         "{We} will keep personal information only for as long as it is needed for these purposes or as required by law.",
                         "{You} can ask for a copy of the personal information {we} hold by writing to {our} data protection officer."]),
    ("Sanctions", ["{We} will not provide cover or pay any claim to the extent that doing so would expose {us} to any sanction or restriction under applicable trade or economic sanctions laws."]),
    ("Currency", ["All amounts in this policy are in US dollars.", "Claims for costs incurred in another currency will be converted at the exchange rate on the date the cost was incurred."]),
    ("Assignment", ["{You} cannot transfer {your} interest in this policy to anyone else without {our} written consent."]),
    ("Notices", ["Any notice {we} give under this policy will be sent to the most recent postal or email address {we} hold.",
                 "Notices from {you} should be sent to the address shown in the schedule."]),
    ("Severability", ["If any term of this policy is found to be invalid, the remaining terms will continue in full force."]),
    ("Renewal", ["{We} will write to {you} at least 21 days before the end of the policy period with the renewal terms.",
                 "{We} are not obliged to offer renewal."]),
    ("Premium payment", ["{You} must pay the premium on or before the policy start date or in the monthly instalments shown in the schedule.",
                         "If an instalment is not paid within {n14} days of its due date, {we} may cancel the policy after giving notice."]),
    ("Reasonable precautions", ["{You} must take all reasonable steps to prevent loss or damage and to keep property in good condition.",
                                "{You} must take reasonable steps to recover lost or stolen property."]),
    ("Claims documents", ["{You} must provide, at {your} own expense, original receipts, invoices, reports and any other evidence {we} reasonably ask for.",
                          "{We} may ask for a medical certificate, a police report or a repair estimate, depending on the claim."]),
    ("Cooperation", ["{You} must give {us} all the help and information {we} reasonably need to settle a claim.",
                     "{You} must not admit liability or offer to settle any claim without {our} written permission."]),
    ("Salvage", ["When {we} replace an item, the damaged or recovered item becomes {our} property.",
                 "If stolen property is recovered after {we} have paid a claim, {you} must tell {us} and either return the payment or give {us} the property."]),
    ("Changes in circumstances", ["{You} must tell {us} as soon as reasonably possible about any change in the information shown in the schedule.",
                                  "{We} may then change the premium or the terms of this policy, or cancel it."]),
    ("Interpretation", ["Headings are for convenience only and do not affect the meaning of this policy.",
                        "Words in the singular include the plural and the other way round.",
                        "Words defined in the definitions section have the same meaning wherever they appear in bold or with an initial capital."]),
    ("Contracts (Rights of Third Parties)", ["A person who is not a party to this policy has no right to enforce any of its terms."]),
    ("Arbitration", ["If {we} accept a claim but disagree with {you} about the amount, the difference will be referred to an arbitrator appointed by both parties.",
                     "Where arbitration applies, obtaining an award is a condition precedent to any legal action against {us}."]),
    ("Joint policyholders", ["Where more than one person is named in the schedule, each of them is treated as having knowledge of anything any of them knows.",
                             "A notice sent to one named person is treated as sent to all of them."]),
    ("Geographical limits", ["Cover applies only within the territory shown in the schedule.",
                             "Cover for property temporarily away from its usual place applies only within the same country."]),
    ("Payment of claims", ["{We} will pay claims by bank transfer to an account in {your} name.",
                           "{We} aim to pay accepted claims within {n10} working days of receiving all the documents {we} need."]),
    ("Interest", ["{We} will not pay interest on any claim payment unless the law requires it."]),
    ("Taxes", ["The premium includes any insurance premium tax at the current rate.", "If the tax rate changes, {we} may adjust the premium from the date of the change."]),
    ("Communication", ["{We} will communicate with {you} in English.", "{We} may record telephone calls for training and quality purposes."]),
    ("Customer support", ["{Our} helpline is open from 8 a.m. to 8 p.m. on weekdays and from 9 a.m. to 5 p.m. on Saturdays.",
                          "Outside these hours {you} can leave a message and {we} will call back on the next working day."]),
    ("Recovery of costs", ["If {we} pay a claim that is later found not to be covered, {we} may ask {you} to repay the amount."]),
    ("Waiver", ["If {we} do not enforce any term of this policy, that does not mean {we} have given up the right to enforce it later."]),
    ("Accessibility", ["This document is available in large print, braille or audio on request."]),
    ("How to read this document", ["This document, the schedule and any endorsements form one contract and should be read together.",
                                   "Where the schedule and this document conflict, the schedule applies.",
                                   "Examples given in this document are for illustration only and do not limit the meaning of any term."]),
    ("Duty of fair presentation", ["{You} must answer all questions {we} ask honestly and to the best of {your} knowledge.",
                                   "If {you} give incorrect information deliberately or recklessly, {we} may treat this policy as if it never existed and keep the premium.",
                                   "If {you} give incorrect information carelessly, {we} may reduce any claim payment in proportion to the premium that should have been charged."]),
    ("Electronic documents", ["{We} will send documents by email unless {you} ask for paper copies.",
                              "{You} must tell {us} if {your} email address changes.",
                              "Documents sent by email are treated as received on the day they are sent."]),
    ("Automatic renewal", ["Unless {you} tell {us} otherwise, this policy will renew automatically at the end of the policy period using the payment details {we} hold.",
                           "{You} can switch off automatic renewal at any time by contacting {our} helpline."]),
    ("Inflation", ["{We} may adjust limits at renewal in line with a published index of consumer prices.",
                   "Any adjustment will be shown in {your} renewal documents; limits do not change during a policy period."]),
    ("Third-party administrators", ["{We} may appoint another company to handle claims or other services on {our} behalf.",
                                    "Any company {we} appoint must follow the terms of this policy and applicable data protection law."]),
    ("Compensation scheme", ["{We} are covered by a financial services compensation scheme.",
                             "If {we} cannot meet {our} obligations, {you} may be entitled to compensation from the scheme, depending on the type of policy and the circumstances."]),
    ("Change of address", ["{You} must tell {us} within 30 days if {you} change {your} address.",
                           "{We} may change the premium or the terms from the date of the move."]),
    ("Discounts", ["Any discount shown in the schedule applies only while the conditions for it continue to be met.",
                   "If a condition for a discount stops being met, {we} may charge the premium without the discount from that date."]),
    ("Instalment agreements", ["If {you} pay by instalments, the instalment agreement is a separate contract with the finance provider.",
                               "Ending the instalment agreement does not cancel this policy, but unpaid premium remains due."]),
    ("Records and evidence", ["{You} should keep receipts, photographs and valuations for items of significant value.",
                              "Where proof of value is missing, {we} may base any settlement on the price of a comparable item.",
                              "{We} may ask for evidence of how and when an item was acquired."]),
    ("Exchange of information", ["To prevent fraud, {we} may check the details {you} give against public records and industry databases.",
                                 "{We} may contact previous insurers to confirm {your} claims history."]),
    ("Market conditions", ["If the replacement item is no longer available, {we} will pay the cost of the nearest equivalent available at the time of the claim."]),
    ("Environmental commitment", ["Where practical, {we} will repair rather than replace damaged items and will use recycled or refurbished parts.",
                                  "{You} may ask for a new part instead, and {we} will tell {you} whether any additional cost would apply."]),
]
# boilerplate that looks relevant to the domain (other covers, services, procedures) but decides nothing in the claim
DOMAIN_FILLER = {
    "travel": [
        ("Emergency assistance", ["{Our} emergency assistance service is available 24 hours a day on the number shown in the schedule.",
                                  "{You} must contact the assistance service before being admitted to hospital as an in-patient, unless it is an emergency.",
                                  "The assistance service can arrange payment guarantees, interpreters and transport home."]),
        ("Repatriation", ["If {our} medical adviser agrees it is necessary, {we} will arrange and pay for {you} to be brought home.",
                          "{We} will decide the method of transport and the time of repatriation after speaking to the treating doctor."]),
        ("Personal liability", ["{We} will pay damages {you} become legally liable to pay for accidental injury to another person or accidental damage to their property during the trip.",
                                "This section does not cover liability arising from the use of motor vehicles, boats or aircraft.",
                                "The most {we} will pay under this section is {n2m} for all claims arising from one event."]),
        ("Missed departure", ["{We} will pay reasonable additional travel and accommodation costs to reach {your} destination if {you} arrive too late to board {your} booked transport because public transport fails.",
                              "{You} must allow enough time to reach the departure point, and must obtain written confirmation of the delay from the transport operator."]),
        ("Rental car excess", ["If {you} hire a car during the trip, {we} will reimburse any excess {you} must pay to the hire company after accidental damage to the car.",
                               "This section applies only to cars hired from a licensed rental company under a written agreement."]),
        ("Legal expenses abroad", ["{We} will pay legal costs to pursue compensation for {your} injury or death caused by a third party during the trip.",
                                   "{We} will not pay legal costs unless {our} legal adviser agrees that the claim has a reasonable prospect of success."]),
    ],
    "home": [
        ("Alternative accommodation", ["If {your} home cannot be lived in after damage covered by this policy, {we} will pay the reasonable cost of comparable temporary accommodation.",
                                       "{We} will also pay for the temporary housing of pets that live with {you}.",
                                       "Cover ends when the home is fit to live in again or after twelve months, whichever is sooner."]),
        ("Locks and keys", ["If the keys to {your} home are lost or stolen, {we} will pay the cost of replacing the locks to the outside doors.",
                            "The most {we} will pay under this clause is {n500} for any one claim."]),
        ("Contents temporarily removed", ["{Your} contents are covered while temporarily removed from the home to another building in the same country, for up to 60 days.",
                                          "Contents in a student hall of residence are covered only if the student is a member of {your} family."]),
        ("Freezer contents", ["{We} will pay for food in a domestic freezer that spoils because of a rise or fall in temperature.",
                              "{We} will not pay if the freezer is more than fifteen years old, or if the electricity supplier cut the supply deliberately."]),
        ("Tenant's improvements", ["If {you} rent the home, {we} will pay for damage to fixtures and fittings that {you} installed and are responsible for.",
                                   "This cover applies only to improvements {you} paid for, not to items provided by the landlord."]),
        ("Garden and outdoor cover", ["{We} will pay for plants, trees and shrubs in the garden of the home that are damaged by fire, lightning or theft.",
                                      "{We} will not pay for damage caused by frost, storms, drought or animals."]),
    ],
    "benefits": [
        ("Eligibility", ["Employees who work at least 20 hours a week are eligible to join the plan from their first day of employment.",
                         "Contractors, agency workers and employees on unpaid leave of more than three months are not eligible."]),
        ("Dependants", ["{You} may enrol dependants at the start of each plan year or within 30 days of a qualifying life event.",
                        "Benefits paid for dependants count towards the same annual limits as {your} own claims."]),
        ("Coordination of benefits", ["If a treatment is also covered by another plan, {we} will pay only the part of the cost that the other plan does not pay.",
                                      "{You} must tell {us} about any other plan that covers {you} or {your} dependants."]),
        ("Appeals", ["If {we} decline all or part of a claim, {you} may appeal in writing within 60 days of the decision.",
                     "An appeal will be reviewed by someone who was not involved in the original decision.",
                     "{We} will give {you} a written decision on the appeal within 30 days."]),
        ("Leaving the plan", ["Cover ends on the day {your} employment ends or the day {you} stop being eligible, whichever is earlier.",
                              "Claims for treatment received before cover ended may be submitted up to 90 days afterwards."]),
        ("Wellness allowance", ["Each plan year {you} may claim up to {n150} towards a gym membership or fitness class.",
                                "The wellness allowance is not subject to any deductible and does not count towards other limits."]),
    ],
    "device": [
        ("Replacement devices", ["If {we} cannot repair {your} device economically, {we} will replace it with a replacement device.",
                                 "Replacement devices may be a different colour and may be refurbished to an as-new standard.",
                                 "Accessories are replaced only if they were lost or damaged in the same incident as the device."]),
        ("Data and backups", ["{We} are not responsible for recovering or restoring any data, software or settings stored on {your} device.",
                              "{You} should back up {your} data regularly and before sending the device for repair."]),
        ("Repair process", ["After a claim is accepted, {we} will send a prepaid mailer or book a courier collection.",
                            "{We} aim to return repaired devices within {n10} business days of receiving them."]),
        ("Transferring the plan", ["{You} may transfer the plan to a new owner of the device by telling {us} in writing.",
                                   "The remaining term of the plan transfers with the device; the term does not restart."]),
        ("Unrecoverable devices", ["If a device is found after {we} have replaced it, the found device becomes {our} property and must be sent to {us}.",
                                   "{We} may block devices reported as lost or stolen so that they cannot be used on mobile networks."]),
        ("Software support", ["{Our} helpline can give advice on setting up and using {your} device.",
                              "Software support does not include fixing problems caused by third-party apps or unofficial operating systems."]),
    ],
}
# the settlement outcome each draw aims for, from a balanced bag (Ctx.pick); deny_exclusion is listed twice, so it is aimed for twice as often
TARGETS = ("pay_less_fee", "pay_sublimit", "pay_limit", "deny_exclusion", "deny_exclusion", "deny_late", "deny_waiting", "not_covered")

# ---------------------------------------------------------------------------------------------------------------- templates
# Six document styles; each fixes heading/clause numbering, voice, how clauses are referenced, how the claim is written and
# the question wording. Templates 4 and 5 are held out of training (TEMPLATE_SPLITS in scripts/build_hard_v1.py).
TEMPLATES = [
    {"voice": "first", "heading": lambda s, t: f"SECTION {s} - {t.upper()}", "clause": lambda s, c, x: f"{s}.{c} {x}",
     "ref": lambda s, c: f"Section {s}.{c}", "case": "form", "date": "us", "state": "text",
     "q1": "Under the policy wording, how should this claim be settled?", "q2": "How much should be paid on this claim?"},
    {"voice": "third", "heading": lambda s, t: f"Article {s}. {t}", "clause": lambda s, c, x: f"({'abcdefghijklmnopqrstuvwxyz'[c - 1]}) {x}",
     "ref": lambda s, c: f"Article {s}({'abcdefghijklmnopqrstuvwxyz'[c - 1]})", "case": "note", "date": "eu", "state": "dict",
     "q1": "What is the correct decision on this claim under the policy?", "q2": "What amount is payable under the policy?"},
    {"voice": "first", "heading": lambda s, t: f"## {s}. {t}", "clause": lambda s, c, x: f"- **{s}.{c}** {x}",
     "ref": lambda s, c: f"clause {s}.{c}", "case": "email", "date": "us", "state": "text",
     "q1": "Which outcome does the policy require for this claim?", "q2": "What is the correct payment for this claim?"},
    {"voice": "third", "heading": lambda s, t: f"PART {roman(s)}: {t}", "clause": lambda s, c, x: f"{roman(s)}.{c}  {x}",
     "ref": lambda s, c: f"Part {roman(s)}, clause {c}", "case": "dict", "date": "iso", "state": "dict",
     "q1": "Applying the policy terms, what should happen to this claim?", "q2": "Applying the policy terms, how much should be paid?"},
    {"voice": "third", "heading": lambda s, t: f"\u00a7 {s}  {t}", "clause": lambda s, c, x: f"({c}) {x}",
     "ref": lambda s, c: f"\u00a7 {s}({c})", "case": "summary", "date": "eu", "state": "text",
     "q1": "Decide this claim according to the policy. Which outcome applies?", "q2": "Settlement amount due under the policy:"},
    {"voice": "first", "heading": lambda s, t: f"{s}. {t}", "clause": lambda s, c, x: f"{s}.{c}. {x}",
     "ref": lambda s, c: f"paragraph {s}.{c}", "case": "chat", "date": "weekday", "state": "text",
     "q1": "Given the policy and the conversation, what is the right outcome for the claim?", "q2": "Given the policy and the conversation, how much should the customer receive?"},
]


def voice_map(template, dom):
    if template["voice"] == "first":
        return {"we": "we", "We": "We", "our": "our", "Our": "Our", "us": "us", "you": "you", "You": "You", "your": "your", "Your": "Your"}
    ins, hol = f"the {dom['insurer']}", f"the {dom['holder']}"
    return {"we": ins, "We": ins[0].upper() + ins[1:], "our": ins + "'s", "Our": ins[0].upper() + ins[1:] + "'s", "us": ins,
            "you": hol, "You": hol[0].upper() + hol[1:], "your": hol + "'s", "Your": hol[0].upper() + hol[1:] + "'s"}


# present-tense verbs that follow {we}/{you} directly; the third-person voice needs their singular form
THIRD_PERSON = {"hold": "holds", "have": "has", "give": "gives", "are": "is", "pay": "pays", "make": "makes", "ask": "asks", "aim": "aims",
                "tell": "tells", "stop": "stops", "replace": "replaces", "rent": "rents", "remain": "remains", "provide": "provides",
                "need": "needs", "do": "does", "decline": "declines", "change": "changes", "become": "becomes", "arrive": "arrives",
                "appoint": "appoints", "agree": "agrees", "accept": "accepts", "hire": "hires"}


def fill(text, slots):
    if slots.get("you", "you") != "you":
        text = re.sub(r"\{(we|We|you|You)\} (\w+)", lambda m: "{%s} %s" % (m.group(1), THIRD_PERSON.get(m.group(2), m.group(2))), text)
    out = text.format_map(slots)
    return out[0].upper() + out[1:] if out else out


def fmt_thr(ex, x):
    return f"{x:,}" if ex.get("unit") == "," else str(x)


# ---------------------------------------------------------------------------------------------------------------- engine
def condition(ex, case):
    """Whether an exclusion's condition holds for the claim (a threshold exclusion compares the stated value; a value the
    claim does not state, or a flag it does not assert, does not trigger the exclusion)."""
    key = ex["key"]
    if ex["kind"] == "flag":
        return bool(case["conditions"].get(key, False))
    v = case["values"].get(key)
    if v is None: return False
    thr = ex["thr"]
    return {"gt": v > thr, "le": v <= thr, "lt": v < thr}[ex["cmp"]]


def declines(f):
    """Every reason the policy gives to decline this claim, as outcome keys."""
    case, cat = f["case"], f["categories"][f["case"]["category"]]
    out = []
    if not cat["purchased"]: out.append("not_covered")
    start, incident, reported = (date.fromisoformat(case[k]) for k in ("start", "incident", "reported"))
    if (incident - start).days < cat["waiting"]: out.append("deny_waiting")
    for ex in f["exclusions"]:
        if case["category"] in ex["categories"] and condition(ex, case) and not (ex["has_exception"] and case["exceptions"].get(ex["key"], False)):
            out.append(f"deny_{ex['key']}")
    if (reported - incident).days > f["notice_days"]: out.append("deny_late")
    return out


def settle(f):
    """(outcome key, amount payable in cents) for a claim no reason declines."""
    case, cat = f["case"], f["categories"][f["case"]["category"]]
    loss, fee, limit = case["loss_cents"], 100 * cat["fee"], 100 * cat["limit"]
    cls = f["classes"].get(case["item_class"]) if case["item_class"] else None
    cap = min(limit, 100 * cls["cap"]) if cls else limit
    if f["fee_order"] == "before":
        amount, binds = min(loss - fee, cap), loss - fee > cap
    else:
        amount, binds = min(loss, cap) - fee, loss > cap
    amount = max(0, amount)
    if not binds: return "pay_less_fee", amount
    return ("pay_sublimit" if cls and 100 * cls["cap"] < limit else "pay_limit"), amount


def solve(f):
    reasons = declines(f)
    if len(reasons) > 1: raise ValueError(f"ambiguous claim: {reasons}")
    outcome, amount = (reasons[0], 0) if reasons else settle(f)
    return {"outcome": outcome, "amount": amount}


# ---------------------------------------------------------------------------------------------------------------- generator
def generate(ctx, t):
    """One long_policy record for template t, or None when a draw is inconsistent (the caller draws again)."""
    rng, tpl = ctx.rng, TEMPLATES[t]
    dom_key = ctx.pick("lp_domain", sorted(DOMAINS))
    dom = DOMAINS[dom_key]
    target = ctx.pick("lp_target", TARGETS)
    issuer = rng.choice(dom["issuers"])
    V = voice_map(tpl, dom)
    # policy parameters
    cats = {}
    for key, c in dom["categories"].items():
        limit = rng.choice(c["limit"])
        cats[key] = {"limit": limit, "fee": rng.choice(c["fee"]), "waiting": rng.choice(c["waiting"]), "optional": bool(c.get("optional")), "purchased": True}
    classes = {}
    for key, c in dom["categories"].items():
        for ck, (title, members, caps) in c["classes"].items():
            caps_ok = [x for x in caps if x < cats[key]["limit"]]
            classes[ck] = {"category": key, "cap": rng.choice(caps_ok), "title": title, "members": members}
    notice = rng.choice([14, 21, 30, 60, 90])
    fee_order = rng.choice(["before", "after"])
    exclusions = []
    thr_values = {}
    for key, c in dom["categories"].items():
        for ex in c["exclusions"]:
            thr = rng.choice(ex["thr"]) if ex["kind"] == "threshold" else None
            if thr is not None: thr_values[ex["key"]] = thr
            if any(e["key"] == ex["key"] for e in exclusions):   # a shared exclusion (unattended) applies to both categories
                next(e for e in exclusions if e["key"] == ex["key"])["categories"].append(key); continue
            exclusions.append({"key": ex["key"], "categories": [key], "kind": ex["kind"], "thr": thr, "cmp": ex.get("cmp"),
                               "has_exception": "exception" in ex})
    spec = {ex["key"]: ex for c in dom["categories"].values() for ex in c["exclusions"]}

    # the claim: category and item for the target outcome
    def cat_ok(key):
        c = dom["categories"][key]
        if target == "not_covered": return c.get("optional", False)
        if target == "deny_waiting": return cats[key]["waiting"] > 0
        if target == "pay_sublimit": return any(i[2] for i in c["items"])
        if target == "pay_limit": return any(i[2] is None for i in c["items"])
        if target == "deny_exclusion": return bool(c["exclusions"])
        return True
    choices = [k for k in dom["categories"] if cat_ok(k)]
    cat = rng.choice(choices)
    cspec = dom["categories"][cat]
    if cats[cat]["optional"]:
        cats[cat]["purchased"] = target != "not_covered"
    for k, c in cats.items():
        if k != cat and c["optional"]: c["purchased"] = rng.random() < 0.5
    items = cspec["items"]
    if target == "pay_sublimit": items = [i for i in items if i[2]]
    if target == "pay_limit": items = [i for i in items if i[2] is None]
    a_item, the_item, item_class = rng.choice(items)
    compatible = [ex for ex in cspec["exclusions"] if not ex.get("only") or the_item in ex["only"]]
    conditions, values, exceptions, sentences = {}, {}, {}, []
    claimant = person(rng)
    first_name = claimant.split()[0]
    slots = {**V, "First": first_name, "first": first_name, "item": the_item, "item_cap": the_item[0].upper() + the_item[1:], "a_item": a_item,
             "extension": dom["extension"], "fee": dom["fee"], "event": dom["event"]}
    denier = tempt = None
    inherent = [ex for ex in compatible if ex.get("only")]   # the item itself matches the exclusion's wording (a veneer, sunglasses)
    if target == "deny_exclusion":
        if not compatible: return None
        denier = inherent[0] if inherent else rng.choice(compatible)
    elif inherent:
        if "exception" not in inherent[0]: return None      # nothing could save the claim: only usable as a decline
        tempt = inherent[0]
    rest = [ex for ex in compatible if ex is not denier]
    if tempt is None and rest and rng.random() < (0.35 if denier else 0.65):
        # beside a declining exclusion only a threshold near miss (a stated number), so two stories never contradict each other
        pool = [ex for ex in rest if ex["kind"] == "threshold" or ("exception" in ex and not denier)]
        tempt = rng.choice(pool) if pool else None
    story_told = denier is not None or (tempt is not None and "exception" in tempt)

    def threshold_value(ex, holds, near):
        """A value on the side of the threshold that makes the condition hold (or not), never equal to it; `near` keeps
        it within two gaps of the threshold."""
        thr, gap = thr_values[ex["key"]], ex.get("gap", 1)
        up = (ex["cmp"] == "gt") == holds          # which side of the threshold the value must be on
        hi = 2 * gap if near else max(3 * gap, thr // 3)
        if not up: hi = min(hi, thr - 1)            # values stay positive
        lo = min(gap, hi)
        d = rng.randint(lo, hi)
        return thr + d if up else thr - d

    for ex in compatible:
        key = ex["key"]
        if ex is denier:
            if ex["kind"] == "flag":
                conditions[key] = True; sentences.append(rng.choice(ex["yes"]))
            else:
                values[key] = threshold_value(ex, True, rng.random() < 0.4)
            if "exception" in ex:
                exceptions[key] = False
                if ex.get("exc_no") and rng.random() < 0.7: sentences.append(rng.choice(ex["exc_no"]))
        elif ex is tempt:
            if "exception" in ex:   # condition holds but the exception saves the claim
                if ex["kind"] == "flag":
                    conditions[key] = True; sentences.append(rng.choice(ex["yes"]))
                else:
                    values[key] = threshold_value(ex, True, False)
                exceptions[key] = True; sentences.append(rng.choice(ex["exc_yes"]))
            else:                   # threshold near miss
                values[key] = threshold_value(ex, False, True)
        else:
            if ex["kind"] == "flag":
                if ex.get("no") and not story_told and rng.random() < 0.4: conditions[key] = False; sentences.append(rng.choice(ex["no"]))
            elif rng.random() < 0.5:
                values[key] = threshold_value(ex, False, False)
    for key, v in values.items():
        sentences.append(spec[key]["value"].replace("{v}", fmt_thr(spec[key], v)))
    if not story_told:   # a benign cause only when no exclusion's story is being told
        sentences.insert(0, rng.choice(cspec["benign"]))
    # dates
    start = date(2025, 1, 1) + timedelta(days=rng.randint(0, 640))
    w = cats[cat]["waiting"]
    if target == "deny_waiting": k = rng.randint(1, w - 1)
    elif w and rng.random() < 0.35: k = w + rng.randint(1, 4)
    else: k = rng.randint(max(w + 1, 2), 330)
    incident = start + timedelta(days=k)
    if target == "deny_late": r = notice + (rng.randint(1, 3) if rng.random() < 0.5 else rng.randint(4, 45))
    elif rng.random() < 0.3: r = notice - rng.randint(1, 3)
    else: r = rng.randint(0, notice - 1)
    reported = incident + timedelta(days=r)
    # amount
    limit, fee = cats[cat]["limit"], cats[cat]["fee"]
    cls = classes.get(item_class) if item_class else None
    cap = min(limit, cls["cap"]) if cls else limit
    if target == "pay_less_fee": loss = rng.randint(fee + 20, max(fee + 21, cap - 20))
    elif target == "pay_sublimit": loss = rng.randint(cap + fee + 30, cap + fee + max(200, 2 * cap))
    elif target == "pay_limit": loss = rng.randint(limit + fee + 50, int(limit * 1.5) + fee + 50)
    else: loss = rng.randint(fee + 50, int(1.3 * cap) + fee)
    loss_cents = 100 * loss + (rng.choice([0, 25, 50, 75, 99, 40]) if rng.random() < 0.3 else 0)
    if target == "pay_less_fee" and loss_cents > 100 * cap: return None

    ex_facts = [{**e, "thr": thr_values.get(e["key"])} for e in exclusions]
    facts = {"domain": dom_key, "fee_order": fee_order, "notice_days": notice,
             "categories": {k: {x: c[x] for x in ("limit", "fee", "waiting", "purchased")} for k, c in cats.items()},
             "classes": {k: {"category": c["category"], "cap": c["cap"]} for k, c in classes.items()},
             "exclusions": ex_facts,
             "case": {"category": cat, "item": the_item, "item_class": item_class, "loss_cents": loss_cents, "start": start.isoformat(),
                      "incident": incident.isoformat(), "reported": reported.isoformat(), "conditions": conditions, "values": values,
                      "exceptions": exceptions}}
    try:
        truth = solve(facts)
    except ValueError:
        return None
    want = f"deny_{denier['key']}" if denier else target
    if truth["outcome"] != want: return None

    # ---- document: a length target in tokens (about 4 characters per token); a short document keeps only the sections
    # the claim and its options refer to, a long one every cover plus boilerplate up to the target
    target_tokens = rng.uniform(850, 4900)
    short = target_tokens < 2000
    wait_cat = cat if cats[cat]["waiting"] else next((k for k in dom["categories"] if cats[k]["waiting"]), None)
    keep = [k for k in dom["categories"] if not short or k == cat or rng.random() < 0.3]
    fee_word = dom["fee"]
    s_fee = {**V, "fee": fee_word, "event": dom["event"], "extension": dom["extension"]}
    sections = []   # (id, title, [clause texts], {clause tag: clause index})
    defs = []
    for ck, c in classes.items():
        if c["category"] in keep: defs.append((c["title"], f"{c['members']}"))
    for key, thr in thr_values.items():
        ex = spec[key]
        if any(k in keep for k in next(e["categories"] for e in exclusions if e["key"] == key)):
            defs.append((ex["term"], ex["definition"].replace("{thr}", fmt_thr(ex, thr))))
    extra_pool = EXTRA_DEFINITIONS + DOMAIN_DEFINITIONS[dom_key]
    defs += rng.sample(extra_pool, rng.randint(1, 3) if short else rng.randint(6, len(extra_pool)))
    defs.sort(key=lambda d: d[0])
    sections.append(("definitions", "Definitions", [fill(f"{term} means {text}.", s_fee) for term, text in defs], {}))
    limits = []
    for key in dom["categories"]:
        c = cats[key]
        limits.append(f"{dom['categories'][key]['title']}: limit {dollars(c['limit'])} per {dom['event']}; {fee_word} {dollars(c['fee'])} per claim.")
    limits.append(fill("The {fee} is deducted from the assessed loss before any limit or sublimit is applied." if fee_order == "before" else
                       "Any limit or sublimit is applied to the assessed loss first, and the {fee} is then deducted from the result.", s_fee))
    sections.append(("limits", f"Limits and {fee_word}", limits, {"order": len(limits)}))
    cat_keys = list(keep); rng.shuffle(cat_keys)
    for key in cat_keys:
        c = dom["categories"][key]
        clauses, tags = [fill(c["cover"], s_fee)], {}
        if c.get("optional"):
            clauses.append(fill("This section applies only if it is shown as purchased in {your} schedule.", s_fee)); tags["optional"] = len(clauses)
        if cats[key]["waiting"]:
            clauses.append(fill(f"No cover applies under this section to any {dom['event']} that happens within the first {cats[key]['waiting']} days after the policy start date.", s_fee))
            tags["waiting"] = len(clauses)
        for ck, (title, _, _) in c["classes"].items():
            clauses.append(fill(f"The most {{we}} will pay for {title} under this section is {dollars(classes[ck]['cap'])} per {dom['event']}.", s_fee))
            tags[f"class:{ck}"] = len(clauses)
        for ex in c["exclusions"]:
            text = fill(f"{{We}} will not pay for {ex['text']}.", s_fee)
            if "exception" in ex: text += " " + fill(ex["exception"], s_fee) + "."
            clauses.append(text); tags[f"ex:{ex['key']}"] = len(clauses)
        sections.append((f"cat:{key}", ("Optional cover: " if c.get("optional") else "") + c["title"], clauses, tags))
    sections.append(("general_exclusions", "General exclusions", [fill(f"{{We}} will not pay for {x}.", s_fee) for x in rng.sample(GENERAL_EXCLUSIONS, rng.randint(2, 3) if short else rng.randint(3, 6))], {}))
    claims = [fill(f"{{You}} must notify {{us}} of a claim within {notice} days of the {dom['event']}. {{We}} will not pay any claim notified after that time.", s_fee),
              fill("{You} must provide proof of ownership or of the cost incurred for every item or service claimed.", s_fee),
              fill("{We} may appoint a loss adjuster to investigate any claim.", s_fee)]
    rng.shuffle(claims)
    sections.append(("claims", "Making a claim", claims, {"notice": claims.index(next(x for x in claims if f"within {notice} days" in x)) + 1}))
    target_chars = int(target_tokens * 4.0)
    filler = rng.sample(FILLER + DOMAIN_FILLER[dom_key], len(FILLER) + len(DOMAIN_FILLER[dom_key]))
    nums = {"n25": dollars(rng.choice([15, 25, 30])), "n14": rng.choice([7, 14, 21]), "n10": rng.choice([5, 10, 15]),
            "n2m": dollars(rng.choice([1000000, 2000000, 5000000])), "n500": dollars(rng.choice([300, 500, 750])), "n150": dollars(rng.choice([100, 150, 250]))}
    body_len = lambda: sum(len(x) + 8 for _, _, cl, _ in sections for x in cl) + 40 * len(sections)
    chosen = []
    while filler and body_len() + sum(len(x) for _, cl in chosen for x in cl) < target_chars:
        title, cl = filler.pop()
        chosen.append((title, [fill(x, {**s_fee, **nums}) for x in cl]))
    order = rng.choice(["filler_last", "filler_mixed"]) if t != 0 else "filler_last"
    head, cat_secs, tail = sections[:2], [s for s in sections if s[0].startswith("cat:")], [s for s in sections[2:] if not s[0].startswith("cat:")]
    fill_secs = [(f"filler:{i}", title, cl, {}) for i, (title, cl) in enumerate(chosen)]
    if t in (1, 3):   # these styles put limits after the covers
        head, cat_secs = head[:1], [head[1]] + cat_secs
    if order == "filler_mixed":
        body = head + cat_secs + tail
        for s in fill_secs: body.insert(rng.randint(2, len(body)), s)
    else:
        body = head + cat_secs + tail + fill_secs
    number = {sid: i + 1 for i, (sid, _, _, _) in enumerate(body)}
    lines = [rng.choice(dom["titles"]).upper() if t != 2 else "# " + rng.choice(dom["titles"]), f"Issued by {issuer}", ""]
    if t in (1, 4):
        lines.insert(2, f"In this policy, \"the {dom['insurer']}\" means {issuer} and \"the {dom['holder']}\" means the person named in the schedule.")
    for sid, title, cl, _ in body:
        s = number[sid]
        lines.append(tpl["heading"](s, title))
        lines += [tpl["clause"](s, i + 1, x) for i, x in enumerate(cl)]
        lines.append("")
    document = "\n".join(lines).strip()
    tags = {sid: tg for sid, _, _, tg in body}
    ref = lambda sid, tag: tpl["ref"](number[sid], tags[sid][tag])

    # ---- claim
    ds = tpl["date"]
    covers = [dom["categories"][k]["title"] for k in dom["categories"] if cats[k]["purchased"]]
    extension_bought = exceptions.get("hazardous") is True
    sched_ext = dom["extension"] if extension_bought else "none"
    rng.shuffle(sentences)
    detail = [fill(d, slots) for d in rng.sample(dom["details"], rng.randint(1, 3))]
    story = [fill(s, slots) for s in sentences]
    if the_item.endswith("s") and not the_item.endswith("ss") or " and " in the_item:   # plural items: agree the verbs that follow them
        for x, y in ((" was ", " were "), (" still works ", " still work "), (" has ", " have ")):
            story = [s.replace(the_item + x, the_item + y).replace(slots["item_cap"] + x, slots["item_cap"] + y) for s in story]
    polno = f"{rng.choice('HPTDBK')}{rng.randint(100000, 999999)}"
    amount_line = money(loss_cents, "whole_ok")
    if tpl["case"] == "form":
        claim = "\n".join([
            "CLAIM FORM", f"Claimant: {claimant}", f"Policy number: {polno}", f"Policy start date: {day(start, ds)}",
            f"Covers shown in schedule: {', '.join(covers)}", f"Extensions shown in schedule: {sched_ext}",
            f"Claim type: {cspec['title']}", f"Item or service claimed: {a_item}", f"Date of {dom['event']}: {day(incident, ds)}",
            f"Date claim notified: {day(reported, ds)}", f"Assessed loss: {amount_line}", "Description: " + " ".join(story + detail)])
    elif tpl["case"] == "note":
        claim = (f"File note. {claimant} (policy {polno}, start date {day(start, ds)}) claims under {cspec['title']} for {a_item}. "
                 f"The schedule shows these covers: {', '.join(covers)}; extensions: {sched_ext}. The {dom['event']} happened on {day(incident, ds)} "
                 f"and was notified on {day(reported, ds)}. " + " ".join(story + detail) + f" The assessed loss is {amount_line}.")
    elif tpl["case"] == "email":
        claim = (f"From: {claimant}\nSubject: Claim on policy {polno}\n\nHello,\n\nI would like to claim under {cspec['title']} for {a_item}. "
                 f"My policy started on {day(start, ds)} and my schedule lists {', '.join(covers)} (extensions: {sched_ext}). "
                 f"The {dom['event']} happened on {day(incident, ds)}. " + " ".join(story + detail) +
                 f"\n\nI am sending this on {day(reported, ds)}, which is when I first notified you. Your adjuster assessed the loss at {amount_line}.\n\nThanks,\n{first_name}")
    elif tpl["case"] == "dict":
        claim = {"claimant": claimant, "policy_number": polno, "schedule": {"start_date": day(start, ds), "covers": covers, "extensions": [dom["extension"]] if extension_bought else []},
                 "claim": {"section": cspec["title"], "item": a_item, f"{dom['event']}_date": day(incident, ds), "notified_date": day(reported, ds),
                           "assessed_loss": amount_line}, "notes": story + detail}
    elif tpl["case"] == "summary":
        claim = (f"Claim summary: {claimant}; policy {polno}; policy start {day(start, ds)}; schedule covers {', '.join(covers)}; extensions {sched_ext}. "
                 f"Section claimed: {cspec['title']}, for {a_item}. {dom['event'].capitalize()} date {day(incident, ds)}; notified {day(reported, ds)}; "
                 f"assessed loss {amount_line}. Facts: " + " ".join(story + detail))
    else:
        claim = "\n".join([f"Customer ({first_name}): Hi, I need to make a claim on policy {polno} under {cspec['title']} for {a_item}.",
                           "Agent: Sure. When did the policy start and what covers does your schedule show?",
                           f"Customer ({first_name}): It started on {day(start, ds)}. The schedule lists {', '.join(covers)}, and extensions: {sched_ext}.",
                           f"Agent: When did the {dom['event']} happen, and what happened?",
                           f"Customer ({first_name}): On {day(incident, ds)}. " + " ".join(story),
                           "Agent: Anything else we should know?", f"Customer ({first_name}): " + (" ".join(detail) or "No."),
                           f"Agent: Noted. I am logging this claim today, {day(reported, ds)}, the first time it has been reported to us. Our adjuster has assessed the loss at {amount_line}."])
    state = {"policy": document, "claim": claim} if tpl["state"] == "dict" else f"{document}\n\n=== CLAIM ===\n{claim if isinstance(claim, str) else claim}"

    # ---- questions
    cls_for_option = item_class if item_class else next(iter(cspec["classes"]), None)
    ex_title = lambda key: spec[key]["title"]
    desc = {"pay_less_fee": f"Pay the assessed loss less the {fee_word}",
            "pay_limit": f"Pay up to the {cspec['title']} limit",
            "deny_late": f"Decline: the claim was notified too late ({ref('claims', 'notice')})",
            "not_covered": f"Decline: {cspec['title']} is not a cover shown in the schedule"}
    if wait_cat in keep: desc["deny_waiting"] = f"Decline: the {dom['event']} happened during the waiting period ({ref('cat:' + wait_cat, 'waiting')})"
    if cls_for_option: desc["pay_sublimit"] = f"Pay up to the {classes[cls_for_option]['title']} sublimit ({ref('cat:' + cat, 'class:' + cls_for_option)})"
    for ex in cspec["exclusions"]:
        desc[f"deny_{ex['key']}"] = f"Decline under the {ex_title(ex['key']).lower()} exclusion ({ref('cat:' + cat, 'ex:' + ex['key'])})"
    correct = truth["outcome"]
    if correct not in desc: return None
    picks = [correct]
    if tempt and f"deny_{tempt['key']}" in desc and f"deny_{tempt['key']}" != correct: picks.append(f"deny_{tempt['key']}")
    pool = [k for k in desc if k not in picks]
    rng.shuffle(pool)
    picks += pool[:5 - len(picks)]
    q1, o1 = choice_q(ctx, tpl["q1"], {k: desc[k] for k in picks}, correct, SRC)
    fee_c, limit_c = 100 * fee, 100 * limit
    cap_c = 100 * cap
    wrong_order = max(0, min(loss_cents, cap_c) - fee_c) if fee_order == "before" else max(0, min(loss_cents - fee_c, cap_c))
    other_cap = 100 * classes[cls_for_option]["cap"] if cls_for_option else limit_c
    distract = [max(0, loss_cents - fee_c), wrong_order, loss_cents, cap_c, max(0, min(loss_cents, other_cap) - fee_c), 0, max(0, limit_c - fee_c),
                max(0, cap_c - fee_c)]
    rng.shuffle(distract)
    if truth["amount"] != 0: distract = [x for x in distract if x != 0] + [0]   # keep $0 in play as a later distractor
    q2, o2 = value_q(ctx, tpl["q2"], truth["amount"], distract, lambda c: money(c, "whole_ok"), SRC)
    facts["options_q"] = {"outcome": o1, "amount": o2}
    meta = {"subtype": dom_key, "target": target, "category": cat, "tempting": tempt["key"] if tempt else None}
    return {"state": state, "questions": {"outcome": q1, "amount": q2}, "facts": facts, "meta": meta}
