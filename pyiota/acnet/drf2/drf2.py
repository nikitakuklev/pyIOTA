import re

from .device import get_qualified_device, parse_device
from .event import DRF_EVENT, parse_event
from .extra import DRF_EXTRA, parse_extra
from .field import DEFAULT_FIELD_FOR_PROPERTY, DRF_FIELD, get_default_field
from .property import DRF_PROPERTY, DRF_PROPERTY_NAMES, get_default_property, \
    parse_property
from .range import DRF_RANGE, parse_range

# 1=DEVIVE, 2=PROPERTY OR FIELD, 3=RANGE, 4=FIELD, 5=EVENT
# PATTERN_FULL = re.compile("(?i)(.{3,}?)" + "(?:\\.(\\w+))?" + "(\\[[\\d:]*\\]|\\{[\\d:]*\\})?" + \
#                           "(?:\\.(\\w+))?" + "(?:@(.+))?")
PATTERN_FULL = re.compile("(?i)(.{3,}?)" + "(?:\\.(\\w+))?" + "(\\[[\\d:]*\\]|\\{[\\d:]*\\})?" + \
                          "(?:\\.(\\w+))?" + "(?:@(.+))?" + '$')


class DiscreteRequest:
    def __init__(self,
                 raw_string: str,
                 device: str,
                 property: DRF_PROPERTY,
                 range: DRF_RANGE,
                 field: DRF_FIELD,
                 event: DRF_EVENT,
                 extra: DRF_EXTRA = None
                 ):
        self.raw_string = raw_string
        self.device = device
        self.property = property
        self.range = range
        self.field = field
        self.event = event
        self.extra = extra

    def __eq__(self, other):
        return self.device == other.device and self.property == other.property and self.range == \
            other.range and self.field == other.field and self.event == other.event and \
            self.extra == other.extra

    def __str__(self):
        return f'DiscreteRequest {self.raw_string} = {self.device=} {self.property=}' \
               f' {self.range=}' \
               f' {self.field=}' \
               f' {self.event=}' \
               f' {self.extra}'

    def __repr__(self):
        return self.__str__()

    @property
    def is_reading(self):
        return self.property == DRF_PROPERTY.READING

    @property
    def is_setting(self):
        return self.property == DRF_PROPERTY.SETTING

    @property
    def is_status(self):
        return self.property == DRF_PROPERTY.STATUS

    @property
    def is_control(self):
        return self.property == DRF_PROPERTY.CONTROL

    @property
    def extra_type(self):
        if self.extra == 'FTP':
            return DRF_EXTRA.FTP
        else:
            raise Exception(f'Unknown extra {self.extra}')

    @property
    def parts(self):
        return self.device, self.property, self.range, self.field, self.event

    def to_canonical(self, device=None, property=None, range=None,
                     field=None, event=None, extra=None
                     ):
        out = ''
        out += device or self.device
        p = property or self.property
        if p is not None:
            out += '.'
            out += p.name
        r = range or self.range
        if r is not None:
            rs = str(r)
            out += rs
        f = field or self.field
        if f is not None:
            if DEFAULT_FIELD_FOR_PROPERTY[p] == f:
                pass
            else:
                fs = f.name
                out += f'.{fs}'
        e = event or self.event
        if e is not None:
            if e.mode != 'U':
                out += f'@{e.raw_string}'
        ex = extra or self.extra
        if ex is not None:
            out += f'<-{ex.name}'
        return out

    def to_qualified(self, device=None, property=None, range=None,
                     field=None, event=None, extra=None
                     ):
        out = ''
        d = device or self.device
        p = property or self.property
        ds = get_qualified_device(d, p)
        out += ds
        r = range or self.range
        if r is not None:
            rs = str(r)
            out += rs
        f = field or self.field
        if f is not None:
            if DEFAULT_FIELD_FOR_PROPERTY[p] == f:
                pass
            else:
                fs = f.name
                out += f'.{fs}'
        e = event or self.event
        if e is not None:
            if e.mode != 'U':
                out += f'@{e.raw_string}'
        ex = extra or self.extra
        if ex is not None:
            out += f'<-{ex.name}'
        return out

    def name_as(self, property: DRF_PROPERTY):
        return get_qualified_device(self.device, property)


def parse_request(device_str: str):
    assert device_str is not None
    if '<-' in device_str:
        splits = device_str.split('<-')
        assert len(splits) == 2, f'Invalid drf {device_str}'
        device_str, extra = splits
        extra_obj = parse_extra(extra)
    else:
        extra_obj = None
    match = PATTERN_FULL.match(device_str)
    if match is None:
        raise ValueError(f'{device_str} is not a valid DRF2 device')
    dev, prop, rng, field, event = match.groups()
    # print(dev, prop, rng, field, event)
    dev_ovj = parse_device(dev)
    dev_obj = dev_ovj.canonical_string
    if prop is None:
        prop_obj = get_default_property(device_str)
    elif prop.upper() in DRF_PROPERTY_NAMES:
        prop_obj = parse_property(prop)
    else:
        prop_obj = get_default_property(device_str)
        field = prop
    rng = parse_range(rng)
    if field is None:
        field = get_default_field(prop_obj)
    event_obj = parse_event(event)
    req = DiscreteRequest(device_str, dev_obj, prop_obj, rng, field, event_obj, extra_obj)
    return req

# class DeviceParser:
#     pattern = re.compile("(?i)[A-Z0][:?_|&@$~][A-Z0-9_:]{1,62}")
#
#     @staticmethod
#     def parse(device_str):
#         assert device_str is not None
#         match = DeviceParser.pattern.match(device_str)
#         if match is None:
#             raise ValueError(f'{device_str} is not a valid device')
#         ld = list(device_str)
#         ld[1] = ':'
#         return ''.join(ld)
#
#     @staticmethod
#     def parse_full(device_str):
#         assert device_str is not None
#         match = PATTERN_FULL.match(device_str)
#         if match is None:
#             raise ValueError(f'{device_str} is not a valid DRF2 device')
#         dev, prop, rng, field, event = match.groups()
#         print(dev, prop, rng, field, event)
#         if prop is None:
#             prop = DRF_PROPERTY.get_by_character(device_str[1])
#             if prop is None:
#                 prop = DRF_PROPERTY.READING
#         rng = parse_range(rng)
#         if field is None:
#             field = get_default_field(prop)
#         event_obj = parse_event(event)
#         return (dev, prop, rng, field, event_obj)
#
#     @staticmethod
#     def get_qualified_device(device_str, prop: DRF_PROPERTY):
#         if len(device_str) < 3:
#             raise ValueError(f'{device_str} is too short for device')
#         assert prop in DRF_PROPERTY
#         ext = prop.value
#         ld = list(device_str)
#         ld[1] = ext
#         return ''.join(ld)

# private static final Map<Property, Field> DEFAULT_FIELDS = new EnumMap<Property, Field>(

# 		put(READING, asList(RAW, PRIMARY, SCALED));
# 		put(SETTING, asList(RAW, PRIMARY, SCALED));
# 		put(STATUS,
# 				asList(RAW, ALL, TEXT, EXTENDED_TEXT, ON, READY, REMOTE,
# 						POSITIVE, RAMP));
# 		put(CONTROL, asList((Field) null));
# 		put(ANALOG,
# 				asList(RAW, ALL, TEXT, MIN, MAX, NOM, TOL, RAW_MIN,
# 						RAW_MAX, RAW_NOM, RAW_TOL, ALARM_ENABLE,
# 						ALARM_STATUS, TRIES_NEEDED, TRIES_NOW, ALARM_FTD,
# 						ABORT, ABORT_INHIBIT, FLAGS));
# 		put(DIGITAL,
# 				asList(RAW, ALL, TEXT, NOM, MASK, ALARM_ENABLE,
# 						ALARM_STATUS, TRIES_NEEDED, TRIES_NOW, ALARM_FTD,
# 						ABORT, ABORT_INHIBIT, FLAGS));
# 		put(DESCRIPTION, asList((Field) null));
# 		put(INDEX, asList((Field) null));
# 		put(LONG_NAME, asList((Field) null));
# 		put(ALARM_LIST_NAME, asList((Field) null));

#     String deviceAttr = m.group(1);
#     String propAttr = m.group(2);
#     String rangeAttr = m.group(3);
#     String fieldAttr = m.group(4);
#     String eventAttr = m.group(5);
#
#     Property prop;
#     if (propAttr == null) {
#     prop = getDefaultProperty(deviceAttr);
#     } else if (rangeAttr != null || fieldAttr != null
#     || Property.isProperty(propAttr)) {
#     prop = Property.parse(propAttr);
#     } else {
#     prop = getDefaultProperty(deviceAttr);
#     fieldAttr = propAttr;
#     }
#
#     Range range = (rangeAttr == null) ? DEFAULT_RANGE : Range \
#     .parse(rangeAttr);
#     Field field = (fieldAttr == null) ? getDefaultField(prop) : Field \
#     .parse(fieldAttr);
#     Event event = (eventAttr == null) ? DEFAULT_EVENT : Event \
#     .parse(eventAttr);
#
#     return new DiscreteRequest(deviceAttr, prop, range, field, event);
#     }
#
#     /**
#     * Gets a default property, based on a <i>device</i> attribute.
#                                                      * <p>
#     * A property qualifier in the given device is used to figure out the
#                                                                      * result.
#                                                                      *
#                                                                      * @param device
#                                                                               *            A <i>device</i> attribute from the same data request; not
#     *            <code>null</code>.
#     * @return A default property.
#     * @throws DeviceFormatException
#     *             If the device is invalid.
#     */
#     public static Property getDefaultProperty(String device) {
#     if (device == null) {
#         throw new NullPointerException();
#     }
#     if (device.length() > 2) {
#     Property prop = DEFAULT_PROPS.get(device.charAt(1));
#     if (prop != null) {
#     return prop;
#     }
#     }
#     return READING; // Just assume READING for everything that can be
#     // understood
# }
#
# /**
# * Gets a default field, based on a <i>property</i> attribute.
#                                                 * <p>
# * Each property has its own default field. Some properties don't support \
#                                                               * fields; in that case this method returns <code>null</code>.
# *
# * @param prop
#          *            A <i>property</i> attribute from the same data request; not
# *            <code>null</code>.
# * @return A default field for that property or <code>null</code>.
# */
# public static Field getDefaultField(Property prop) {
# if (prop == null) {
#     throw new NullPointerException();
# }
# return DEFAULT_FIELDS.get(prop);
# }
#
# /**
# * Creates the request object from a set of individual attrubutes.
#                                                       * <p>
# * Each attribute must be valid as per DRF2, but doesn't need to be
#                                                      * canonical. The entire combination of attributes must also be valid; in
# * particular, the <i>field</i> must be permitted within the
#                                                         * <i>property</i>.
# * <p>
# * To create a request object from a text string, use the
#                                                      * {@link DataRequest#parse(String)} static method.
# *
# * @param device
# *            a <i>device</i> attribute; not <code>null</code>.
# * @param property
# *            a <i>property</i> attribute; not <code>null</code>.
# * @param range
# *            a <i>range</i> attribute; not <code>null</code>.
# * @param field
# *            a <i>field</i> attribute; may be <code>null</code> only if the
# *            property doesn't support fields.
# * @param event
# *            an <i>event</i> attribute; not <code>null</code>.
# * @throws RequestFormatException
# *             if the attributes are invalid.
# */
# public DiscreteRequest(String device, Property property, Range range,
# Field field, Event event) throws RequestFormatException,
# IllegalArgumentException {
# if (device == null || property == null || range == null
#     || event == null) {
#     throw new NullPointerException();
# }
# if (device.charAt(1) != ':' && property != getDefaultProperty(device)) {
# throw new IllegalArgumentException("Property mismatch");
# }
# if (!PROP_FIELDS.get(property).contains(field)) {
# throw new IllegalArgumentException("Illegal field");
# }
# this.device = DeviceParser.parse(device);
# this.deviceUC = this.device.toUpperCase();
# this.property = property;
# this.range = range;
# this.field = field;
# this.event = event;
# }
#
# /**
# * Gets the <i>device</i> attribute of this request in a canonical form.
# *
# * @return The <i>device</i> attribute, not <code>null</code>.
# */
# public String getDevice() {
# return device;
# }
#
# /**
# * Gets the <i>property</i> attribute of this request.
#                                              *
#                                              * @return The <i>property</i> attribute, not <code>null</code>.
# */
# public Property getProperty() {
# return property;
# }
#
# /**
# * Gets the <i>range</i> attribute of this request.
#                                           * <p>
# * In the case of a default range, which doesn't appear in the canonical
#                                              * request string, this method returns that default value.
#                                                                                                 *
#                                                                                                 * @return The <i>range</i> attribute, not <code>null</code>.
# */
# public Range getRange() {
# return range;
# }
#
# /**
# * Gets the <i>field</i> attribute of this request.
#                                           * <p>
# * In the case of a default field, which doesn't appear in the canonical
#                                              * request string, this method returns that default value. If the property
#                                                                                                               * doesn't support fields, this method returns <code>null</code>.
#                                                                                                                      *
#                                                                                                                      * @return The <i>field</i> attribute, or <code>null</code> if the property
# *         doesn't support fields.
# */
# public Field getField() {
# return field;
# }
#
# /**
# * Gets the <i>event</i> attribute of this request in a canonical form.
#                                                                  *
#                                                                  * @return The <i>event</i> attribute, not <code>null</code>.
# */
# public Event getEvent() {
# return event;
# }
#
# @Override
# public int hashCode() {
# return deviceUC.hashCode() ^ property.hashCode() ^ range.hashCode()
# ^ ((field == null) ? 0 : field.hashCode()) ^ event.hashCode();
#
# }
#
# @Override
# public boolean equals(Object obj) {
# return (obj instanceof DiscreteRequest)
# && ((DiscreteRequest) obj).deviceUC.equals(deviceUC)
# && ((DiscreteRequest) obj).property == property
# && ((DiscreteRequest) obj).range.equals(range)
# && (((DiscreteRequest) obj).field != null ? ((DiscreteRequest) obj).field
# .equals(field) : field == null)
# && ((DiscreteRequest) obj).event.equals(event);
# }
#
# /**
# * Returns the canonical textual form of this request.
#                                              * <p>
# * This method is a convenient way to call
#                                      * {@link DataRequestParser#format(DataRequest)}.
# *
# * @return Canonical data request.
# * @see DataRequestParser
# */
# @Override
# public String toString() {
# if (text == null) {
#     StringBuilder buf = new StringBuilder();
# buf.append(device);
# buf.append('.');
# buf.append(property);
# if (!range.equals(DEFAULT_RANGE)) {
#     buf.append(range);
# }
# if (field != getDefaultField(property)) {
# buf.append('.');
# buf.append(field);
# }
# if (!event.equals(DEFAULT_EVENT)) {
# buf.append('@');
# buf.append(event);
# }
# text = buf.toString();
# }
# return text;
# }
#
# }
