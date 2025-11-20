"""
Logits processor v2 for structured output generation.

This module implements a state machine-based logits processor that guides
LLM generation to produce structured JSON output with item labels (main/other).
"""

from enum import Enum

import torch
from transformers import AutoTokenizer

from dripper.exceptions import DripperLogitsError


def mask_other_logits(logits: torch.Tensor, remained_ids: list[int]):
    """
    Mask all logits except those in remained_ids by setting them to -inf.

    This function restricts the model to only generate tokens with IDs in
    the remained_ids list.

    Args:
        logits: Original logits tensor
        remained_ids: List of token IDs to keep (all others will be masked)

    Returns:
        New logits tensor with only remained_ids having non-negative values
    """
    remained_logits = {ids: logits[ids].item() for ids in remained_ids}
    new_logits = torch.ones_like(logits, dtype=torch.bfloat16) * -float('inf')
    for i, id in enumerate(remained_ids):
        new_logits[id] = remained_logits[id]
    return new_logits


class OuterStateCategory(Enum):
    """
    Outer state categories for the state machine.

    These represent the high-level states in the generation process.
    """

    Begin = 0  # Initial state
    Decide_Main_Other = 2  # Processing item labels (main/other)
    End = 3  # Finished generating JSON structure
    EOS = 5  # End of sequence


class InnerStateCategory(Enum):
    """
    Inner state categories within Decide_Main_Other state.

    These represent sub-states for processing individual item entries.
    """

    WaitingForQuote = 0  # Waiting for opening quote
    BuildingNumber = 1  # Building item ID number
    WaitingForColon = 2  # Waiting for colon (unused in current implementation)
    WaitingForValue = 3  # Waiting for main/other value
    WaitingForCommaOrEnd = 4  # Waiting for comma or closing bracket


class OuterState:
    """
    Outer state representation for the state machine.

    Contains the current outer state category and, if in Decide_Main_Other state,
    tracks the current item ID, number piece being built, and inner state.
    """

    def __init__(
        self,
        category: OuterStateCategory,
        item_id: int = None,
        num_piece: str = None,
        inner_state: InnerStateCategory = None,
    ):
        """
        Initialize OuterState.

        Args:
            category: Outer state category
            item_id: Current item ID (required for Decide_Main_Other)
            num_piece: Current number piece being built (required for Decide_Main_Other)
            inner_state: Inner state category (required for Decide_Main_Other)

        Raises:
            DripperLogitsError: If Decide_Main_Other state is missing required parameters
        """
        self.category = category
        if category == OuterStateCategory.Decide_Main_Other:
            if item_id is None or num_piece is None:
                raise DripperLogitsError(
                    'item_id and num_piece must be provided for Decide_Main_Other state'
                )
            self.item_id = item_id
            self.num_piece = num_piece
            self.inner_state = inner_state or InnerStateCategory.WaitingForQuote
        else:
            self.item_id = None
            self.num_piece = None
            self.inner_state = None

    def __str__(self):
        return (
            f"OuterState(category={self.category}, item_id={self.item_id}, "
            f"num_piece='{self.num_piece}', inner={self.inner_state})"
        )


class SpecialSingleTokens(Enum):
    """
    Special single-token representations for structured generation.

    These tokens represent specific characters or sequences that need to be
    generated as single tokens in the output format.
    """

    EOS = -2  # End of sequence
    D0 = 0  # Digit 0
    D1 = 1  # Digit 1
    D2 = 2  # Digit 2
    D3 = 3  # Digit 3
    D4 = 4  # Digit 4
    D5 = 5  # Digit 5
    D6 = 6  # Digit 6
    D7 = 7  # Digit 7
    D8 = 8  # Digit 8
    D9 = 9  # Digit 9
    Left_bracket = 10  # Opening brace '{'
    Quote_Right_bracket = 11  # Closing quote and brace '"}'
    Quote = 12  # Single quote '"'
    Quote_colon_quote = 13  # Quote, colon, quote '":"'
    Quote_comma = 14  # Quote and comma '",'
    Bracket_pair = 15  # Empty object '{}'
    Main = 19  # String 'main'
    Other = 20  # String 'other'
    Think = 21  # Reasoning tag '</think>'
    NNewline = 22  # Double newline '\n\n'

    @classmethod
    def from_digit(cls, digit: int):
        """
        Create SpecialSingleTokens enum from a digit (0-9).

        Args:
            digit: Integer digit from 0 to 9

        Returns:
            Corresponding SpecialSingleTokens enum value

        Raises:
            DripperLogitsError: If digit is not in range 0-9
        """
        if 0 <= digit <= 9:
            return cls(digit)
        else:
            raise DripperLogitsError(f'digit {digit} is not a valid digit')

    @staticmethod
    def number_set():
        """
        Get set of all digit tokens (D0-D9).

        Returns:
            Frozenset containing all digit token enums
        """
        return frozenset([
            SpecialSingleTokens.D0,
            SpecialSingleTokens.D1,
            SpecialSingleTokens.D2,
            SpecialSingleTokens.D3,
            SpecialSingleTokens.D4,
            SpecialSingleTokens.D5,
            SpecialSingleTokens.D6,
            SpecialSingleTokens.D7,
            SpecialSingleTokens.D8,
            SpecialSingleTokens.D9,
        ])


# Token string mapping: maps SpecialSingleTokens enum to their string representations
special_single_tokens_map = {
    SpecialSingleTokens.EOS: '<|im_end|>',
    SpecialSingleTokens.D0: '0',
    SpecialSingleTokens.D1: '1',
    SpecialSingleTokens.D2: '2',
    SpecialSingleTokens.D3: '3',
    SpecialSingleTokens.D4: '4',
    SpecialSingleTokens.D5: '5',
    SpecialSingleTokens.D6: '6',
    SpecialSingleTokens.D7: '7',
    SpecialSingleTokens.D8: '8',
    SpecialSingleTokens.D9: '9',
    SpecialSingleTokens.Left_bracket: '{',
    SpecialSingleTokens.Quote_Right_bracket: '"}',
    SpecialSingleTokens.Quote: '"',  # Single quote
    SpecialSingleTokens.Quote_colon_quote: '":"',
    SpecialSingleTokens.Quote_comma: '",',
    SpecialSingleTokens.Bracket_pair: '{}',
    SpecialSingleTokens.Main: 'main',
    SpecialSingleTokens.Other: 'other',
    SpecialSingleTokens.Think: '</think>',
    SpecialSingleTokens.NNewline: '\n\n',
}


def get_static_logits(ids_to_remained: int, device: str = 'cuda'):
    """
    Create logits tensor that only allows a specific token.

    Creates a logits tensor where all values are -inf except for the
    specified token ID which is set to 1.0.

    Args:
        ids_to_remained: Token ID to allow
        device: Device to create tensor on ('cuda' or 'cpu')

    Returns:
        Logits tensor with only the specified token enabled
    """
    base_tensor = (
        torch.ones(151936, dtype=torch.bfloat16, device=device)
        * -float('inf')
    )
    base_tensor[ids_to_remained] = 1.0
    return base_tensor


def get_special_logits_map(tokenizer: AutoTokenizer, device: str = 'cuda'):
    """
    Build logits map for special tokens.

    Creates a dictionary mapping SpecialSingleTokens enum values to their
    corresponding logits tensors. Each logits tensor only allows the
    corresponding token to be generated.

    Args:
        tokenizer: Tokenizer to encode token strings
        device: Device to create tensors on ('cuda' or 'cpu')

    Returns:
        Dictionary mapping SpecialSingleTokens to logits tensors
    """
    res_dict = {}
    for k, v in special_single_tokens_map.items():
        token_ids = tokenizer.encode(v, add_special_tokens=False)
        if len(token_ids) == 1:
            res_dict[k] = get_static_logits(token_ids[0], device)
        else:
            # If multiple tokens, use only the first one
            print(
                f'Warning: {k}:{v} encodes to multiple tokens {token_ids}, '
                f'using first one'
            )
            res_dict[k] = get_static_logits(token_ids[0], device)
    return res_dict


class TokenStateMachine:
    """
    State machine for controlling structured token generation.

    This class implements a two-level state machine (outer and inner states)
    to guide LLM generation of structured JSON output with item labels.
    """

    def __init__(self, max_count, tokenizer: AutoTokenizer, device: str = 'cuda'):
        """
        Initialize TokenStateMachine.

        Args:
            max_count: Maximum number of items to process
            tokenizer: Tokenizer for encoding/decoding tokens
            device: Device to use for tensor operations ('cuda' or 'cpu')
        """
        self.max_count = max_count
        self.tokenizer = tokenizer
        self.outer_state = OuterState(OuterStateCategory.Begin)
        self.device = device

        # Build token ID mapping
        self.special_single_tokens_ids_map = {}
        for token, token_str in special_single_tokens_map.items():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                self.special_single_tokens_ids_map[token] = ids[0]
            else:
                print(f'Warning: {token}:{token_str} -> {ids}')
                self.special_single_tokens_ids_map[token] = ids[0]

        # Digit set (0-9)
        self.number_set = frozenset([
            self.special_single_tokens_ids_map[dig]
            for dig in SpecialSingleTokens.number_set()
        ])

        # main/other set
        self.main_other_set = frozenset([
            self.tokenizer.encode('main', add_special_tokens=False)[0],
            self.tokenizer.encode('other', add_special_tokens=False)[0],
        ])

        # Pre-computed logits map
        self.special_logits_map = get_special_logits_map(self.tokenizer, device)

    def handle_begin(
        self, input_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Handle Begin state with Think logic.

        Implements a three-step initialization:
        1. First token: force '</think>'
        2. Second token: force '\n\n'
        3. Third token: start JSON generation

        Args:
            input_ids: List of generated token IDs so far
            logits: Current logits tensor

        Returns:
            Modified logits tensor for next token generation
        """
        # Step 1: If first token, force output '</think>'
        if len(input_ids) == 0:
            return self.special_logits_map[SpecialSingleTokens.Think]

        # Step 2: If second token, force output '\n\n'
        elif len(input_ids) == 1:
            return self.special_logits_map[SpecialSingleTokens.NNewline]

        # Step 3: If third token, start original logic
        elif len(input_ids) == 2:
            if self.max_count == 0:
                # Empty object
                output_logits = self.special_logits_map[
                    SpecialSingleTokens.Bracket_pair
                ]
                self.outer_state = OuterState(OuterStateCategory.End)
            else:
                # Start generating JSON
                output_logits = self.special_logits_map[
                    SpecialSingleTokens.Left_bracket
                ]
                self.outer_state = OuterState(
                    OuterStateCategory.Decide_Main_Other,
                    1,
                    '',
                    InnerStateCategory.WaitingForQuote,
                )
            return output_logits

        # Safety: if state is still Begin but multiple tokens generated, don't intervene
        else:
            return logits

    def handle_end(self, input_ids: list[int], logits: torch.Tensor):
        """
        Handle End state.

        Transitions to EOS state and forces EOS token generation.

        Args:
            input_ids: List of generated token IDs so far
            logits: Current logits tensor

        Returns:
            Logits tensor forcing EOS token
        """
        output_logits = self.special_logits_map[SpecialSingleTokens.EOS]
        self.outer_state = OuterState(OuterStateCategory.EOS)
        return output_logits

    def handle_eos(self, input_ids: list[int], logits: torch.Tensor):
        """
        Handle EOS state.

        Continues to force EOS token generation.

        Args:
            input_ids: List of generated token IDs so far
            logits: Current logits tensor

        Returns:
            Logits tensor forcing EOS token
        """
        return self.special_logits_map[SpecialSingleTokens.EOS]

    def handle_decide_main_other(
        self, input_ids: list[int], logits: torch.Tensor
    ):
        """
        Handle Decide_Main_Other state - core of the two-level state machine.

        Processes the inner states to generate structured JSON output with
        item IDs and main/other labels.

        Args:
            input_ids: List of generated token IDs so far
            logits: Current logits tensor

        Returns:
            Modified logits tensor for next token generation

        Raises:
            DripperLogitsError: If unexpected token is encountered in any inner state
        """
        last_token = input_ids[-1]

        # Process based on inner state
        if self.outer_state.inner_state == InnerStateCategory.WaitingForQuote:
            # Expect quote (after { or ,)
            if (
                last_token
                == self.special_single_tokens_ids_map[
                    SpecialSingleTokens.Left_bracket
                ]
                or last_token
                == self.special_single_tokens_ids_map[
                    SpecialSingleTokens.Quote_comma
                ]
            ):
                output_logits = self.special_logits_map[SpecialSingleTokens.Quote]
                self.outer_state.inner_state = InnerStateCategory.BuildingNumber
                self.outer_state.num_piece = ''
            else:
                raise DripperLogitsError(
                    f'Unexpected token in WaitingForQuote: {last_token}'
                )

        elif self.outer_state.inner_state == InnerStateCategory.BuildingNumber:
            # Building number (item ID)
            if (
                last_token
                == self.special_single_tokens_ids_map[SpecialSingleTokens.Quote]
            ):
                # Just started building number
                target_num_str = str(self.outer_state.item_id)
                next_char = target_num_str[0]
                output_logits = self.special_logits_map[
                    SpecialSingleTokens.from_digit(int(next_char))
                ]
                self.outer_state.num_piece = next_char
            elif last_token in self.number_set:
                # Continue building number
                target_num_str = str(self.outer_state.item_id)
                if len(self.outer_state.num_piece) < len(target_num_str):
                    next_char = target_num_str[len(self.outer_state.num_piece)]
                    output_logits = self.special_logits_map[
                        SpecialSingleTokens.from_digit(int(next_char))
                    ]
                    self.outer_state.num_piece += next_char
                else:
                    # Number building complete, generate ":"
                    output_logits = self.special_logits_map[
                        SpecialSingleTokens.Quote_colon_quote
                    ]
                    self.outer_state.inner_state = InnerStateCategory.WaitingForValue
            else:
                raise DripperLogitsError(
                    f'Unexpected token in BuildingNumber: {last_token}'
                )

        elif self.outer_state.inner_state == InnerStateCategory.WaitingForValue:
            # Waiting for main/other
            if (
                last_token
                == self.special_single_tokens_ids_map[
                    SpecialSingleTokens.Quote_colon_quote
                ]
            ):
                # Let model choose main or other
                output_logits = mask_other_logits(
                    logits, list(self.main_other_set)
                )
                self.outer_state.inner_state = (
                    InnerStateCategory.WaitingForCommaOrEnd
                )
            else:
                raise DripperLogitsError(
                    f'Unexpected token in WaitingForValue: {last_token}'
                )

        elif (
            self.outer_state.inner_state
            == InnerStateCategory.WaitingForCommaOrEnd
        ):
            # After main/other, decide comma or end
            if last_token in self.main_other_set:
                if self.outer_state.item_id == self.max_count:
                    # Last item, generate '"}'
                    output_logits = self.special_logits_map[
                        SpecialSingleTokens.Quote_Right_bracket
                    ]
                    self.outer_state = OuterState(OuterStateCategory.End)
                else:
                    # Not last item, generate '",'
                    output_logits = self.special_logits_map[
                        SpecialSingleTokens.Quote_comma
                    ]
                    self.outer_state = OuterState(
                        OuterStateCategory.Decide_Main_Other,
                        self.outer_state.item_id + 1,
                        '',
                        InnerStateCategory.WaitingForQuote,
                    )
            else:
                raise DripperLogitsError(
                    f'Unexpected token in WaitingForCommaOrEnd: {last_token}'
                )
        else:
            raise DripperLogitsError(
                f'Unknown inner state: {self.outer_state.inner_state}'
            )

        return output_logits

    def process_logit(
        self, input_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Main processing function for logits modification.

        Routes to appropriate handler based on current outer state category.

        Args:
            input_ids: List of generated token IDs so far
            logits: Current logits tensor from the model

        Returns:
            Modified logits tensor for next token generation
        """
        if self.outer_state.category == OuterStateCategory.Begin:
            output_logits = self.handle_begin(input_ids, logits)
        elif self.outer_state.category == OuterStateCategory.End:
            output_logits = self.handle_end(input_ids, logits)
        elif (
            self.outer_state.category
            == OuterStateCategory.Decide_Main_Other
        ):
            output_logits = self.handle_decide_main_other(input_ids, logits)
        elif self.outer_state.category == OuterStateCategory.EOS:
            output_logits = self.handle_eos(input_ids, logits)
        else:
            print(
                f'Unhandled outer state: {self.outer_state.category}'
            )
            output_logits = logits

        return output_logits
