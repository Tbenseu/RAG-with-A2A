# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: web_search_agent.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'web_search_agent.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16web_search_agent.proto\"\x1e\n\rSearchRequest\x12\r\n\x05query\x18\x01 \x01(\t\"\x82\x01\n\x0eSearchResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\x12/\n\x08metadata\x18\x02 \x03(\x0b\x32\x1d.SearchResponse.MetadataEntry\x1a/\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x32=\n\x10WebSearchService\x12)\n\x06Search\x12\x0e.SearchRequest\x1a\x0f.SearchResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'web_search_agent_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SEARCHRESPONSE_METADATAENTRY']._loaded_options = None
  _globals['_SEARCHRESPONSE_METADATAENTRY']._serialized_options = b'8\001'
  _globals['_SEARCHREQUEST']._serialized_start=26
  _globals['_SEARCHREQUEST']._serialized_end=56
  _globals['_SEARCHRESPONSE']._serialized_start=59
  _globals['_SEARCHRESPONSE']._serialized_end=189
  _globals['_SEARCHRESPONSE_METADATAENTRY']._serialized_start=142
  _globals['_SEARCHRESPONSE_METADATAENTRY']._serialized_end=189
  _globals['_WEBSEARCHSERVICE']._serialized_start=191
  _globals['_WEBSEARCHSERVICE']._serialized_end=252
# @@protoc_insertion_point(module_scope)
